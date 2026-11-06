//! Generic bypass proxy handler for unknown API paths
//!
//! This module provides a simple HTTP proxy that forwards requests directly
//! to the backend provider without any normalization or routing logic.
//! Used for paths not in the intercepted list (embeddings, audio, images, etc.)

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use lunaroute_routing::PathClassifier;
use reqwest::Client;
use std::sync::Arc;
use tracing::{debug, error, warn};

use crate::ProviderType;

/// Bypass proxy error
#[derive(Debug, thiserror::Error)]
pub enum BypassError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Invalid header name: {0}")]
    InvalidHeaderName(String),

    #[error("Invalid header value: {0}")]
    InvalidHeaderValue(String),

    #[error("Body read error: {0}")]
    BodyReadError(String),

    #[error("No provider available for bypass")]
    NoProviderAvailable,

    #[error(
        "Cannot determine the upstream provider; include provider-specific headers or a model, or set bypass.provider"
    )]
    AmbiguousProvider,

    #[error("Request path and credentials identify different upstream providers")]
    ConflictingProvider,
}

impl IntoResponse for BypassError {
    fn into_response(self) -> Response {
        let status = match &self {
            BypassError::NoProviderAvailable => StatusCode::SERVICE_UNAVAILABLE,
            BypassError::AmbiguousProvider | BypassError::ConflictingProvider => {
                StatusCode::BAD_REQUEST
            }
            _ => StatusCode::BAD_GATEWAY,
        };

        let body = serde_json::json!({
            "error": "bypass_proxy_error",
            "message": self.to_string(),
        })
        .to_string();

        (status, body).into_response()
    }
}

/// Configuration for a bypass provider (simplified provider info)
#[derive(Debug, Clone)]
pub struct BypassProvider {
    /// Base URL of the provider (e.g., "https://api.openai.com/v1")
    pub base_url: String,
    /// API key for authentication
    pub api_key: String,
    /// Provider name for logging
    pub name: String,
    /// HTTP client
    pub client: Arc<Client>,
    /// Maximum inbound request body size in bytes (mirrors the server-wide
    /// DefaultBodyLimit; used for the bypass handler's own bounded read).
    pub max_request_body_bytes: usize,
}

impl BypassProvider {
    /// Create a new bypass provider
    pub fn new(
        base_url: String,
        api_key: String,
        name: String,
        client: Arc<Client>,
        max_request_body_bytes: usize,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            name,
            client,
            max_request_body_bytes,
        }
    }

    /// Build the full URL for a path
    fn build_url(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/{}", self.base_url, path)
    }
}

/// Wrap an existing router with bypass functionality
///
/// This adds a fallback route that handles unknown paths by proxying them
/// directly to the provider if bypass is enabled.
pub fn with_bypass(
    router: Router,
    provider: Option<Arc<BypassProvider>>,
    classifier: Arc<PathClassifier>,
) -> Router {
    if !classifier.is_bypass_enabled() || provider.is_none() {
        // Bypass disabled, return router as-is
        return router;
    }

    // Clone for closure
    let provider_clone = provider.clone();
    let classifier_clone = classifier.clone();

    // Add fallback handler using a closure that captures the bypass state
    router.fallback(move |req: Request| {
        let provider = provider_clone.clone();
        let classifier = classifier_clone.clone();
        async move { bypass_handler_impl(provider, classifier, req).await }
    })
}

/// Route auxiliary requests by protocol, credentials, and model without changing their bodies.
/// Shared model-list endpoints use the same selection as other bypass requests.
pub fn with_auto_bypass(
    router: Router,
    openai: Option<Arc<BypassProvider>>,
    anthropic: Option<Arc<BypassProvider>>,
    classifier: Arc<PathClassifier>,
) -> Router {
    if !classifier.is_bypass_enabled() || (openai.is_none() && anthropic.is_none()) {
        return router;
    }

    router
        .fallback(|| async { StatusCode::NOT_FOUND })
        .layer(middleware::from_fn(move |req: Request, next: Next| {
            let openai = openai.clone();
            let anthropic = anthropic.clone();
            let classifier = classifier.clone();
            async move {
                let path = req.uri().path();
                // Validate provider-specific credentials before forwarding native requests too.
                let from_headers = match header_provider(req.headers()) {
                    Ok(provider) => provider,
                    Err(error) => return error.into_response(),
                };
                let from_path = path_provider(path);
                if matches!((from_path, from_headers), (Some(a), Some(b)) if a != b) {
                    return BypassError::ConflictingProvider.into_response();
                }
                let models =
                    path == "/v1/models" && matches!(*req.method(), Method::GET | Method::HEAD);
                if classifier.should_bypass(path) || models {
                    match auto_bypass_request(openai, anthropic, req, from_path.or(from_headers))
                        .await
                    {
                        Ok(response) => response,
                        Err(error) => error.into_response(),
                    }
                } else {
                    next.run(req).await
                }
            }
        }))
}

fn path_provider(path: &str) -> Option<ProviderType> {
    let matches = |prefix: &&str| {
        path == *prefix
            || path
                .strip_prefix(*prefix)
                .is_some_and(|tail| tail.starts_with('/'))
    };
    if path == "/api/hello" || ["/v1/messages", "/v1/complete"].iter().any(matches) {
        return Some(ProviderType::Anthropic);
    }
    [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/responses",
        "/v1/embeddings",
        "/v1/audio",
        "/v1/images",
        "/v1/moderations",
        "/v1/realtime",
        "/v1/fine-tuning",
        "/v1/assistants",
        "/v1/threads",
        "/v1/vector_stores",
        "/v1/batches",
        "/v1/uploads",
    ]
    .iter()
    .any(matches)
    .then_some(ProviderType::OpenAI)
}

fn header_provider(headers: &HeaderMap) -> Result<Option<ProviderType>, BypassError> {
    let mut anthropic = headers.keys().any(|name| {
        name.as_str().starts_with("anthropic-") || name.as_str().starts_with("x-claude-code-")
    }) || headers.contains_key("x-api-key");
    let mut openai = headers
        .keys()
        .any(|name| name.as_str().starts_with("openai-"))
        || headers.contains_key("chatgpt-account-id");
    for value in headers.get_all("authorization") {
        if let Ok(value) = value.to_str()
            && let Some((scheme, token)) = value.split_once(' ')
            && scheme.eq_ignore_ascii_case("bearer")
        {
            if token.starts_with("sk-ant-") {
                anthropic = true;
            } else if token.starts_with("sk-") {
                openai = true;
            }
        }
    }
    match (anthropic, openai) {
        (true, true) => Err(BypassError::ConflictingProvider),
        (true, false) => Ok(Some(ProviderType::Anthropic)),
        (false, true) => Ok(Some(ProviderType::OpenAI)),
        (false, false) => Ok(None),
    }
}

async fn auto_bypass_request(
    openai: Option<Arc<BypassProvider>>,
    anthropic: Option<Arc<BypassProvider>>,
    req: Request,
    mut provider_type: Option<ProviderType>,
) -> Result<Response, BypassError> {
    let (parts, body) = req.into_parts();
    let path = parts.uri.path();
    // A known protocol must not fall back to a different provider when it is disabled.
    let limit = match provider_type {
        Some(ProviderType::OpenAI) => openai
            .as_ref()
            .map(|provider| provider.max_request_body_bytes),
        Some(ProviderType::Anthropic) => anthropic
            .as_ref()
            .map(|provider| provider.max_request_body_bytes),
        None => openai
            .iter()
            .chain(anthropic.iter())
            .map(|provider| provider.max_request_body_bytes)
            .max(),
    }
    .ok_or(BypassError::NoProviderAvailable)?;
    let body = axum::body::to_bytes(body, limit)
        .await
        .map_err(|error| BypassError::BodyReadError(error.to_string()))?;

    if provider_type.is_none()
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body)
        && let Some(model) = json.get("model").and_then(|model| model.as_str())
    {
        provider_type = if model.starts_with("claude-") {
            Some(ProviderType::Anthropic)
        } else if model.starts_with("gpt-")
            || model.starts_with("chatgpt-")
            || model.starts_with("text-embedding-")
            || ["o1", "o3", "o4"]
                .iter()
                .any(|prefix| model.starts_with(prefix))
        {
            Some(ProviderType::OpenAI)
        } else {
            None
        };
    }
    let provider = match provider_type {
        Some(ProviderType::OpenAI) => openai,
        Some(ProviderType::Anthropic) => anthropic,
        None => match (openai, anthropic) {
            (Some(provider), None) | (None, Some(provider)) => Some(provider),
            _ => return Err(BypassError::AmbiguousProvider),
        },
    }
    .ok_or(BypassError::NoProviderAvailable)?;
    if body.len() > provider.max_request_body_bytes {
        return Err(BypassError::BodyReadError(
            "Request body exceeds provider limit".into(),
        ));
    }
    let path = parts
        .uri
        .path_and_query()
        .map(|uri| uri.as_str())
        .unwrap_or(path);
    proxy_request(&provider, path, parts.method, parts.headers, body).await
}

/// Implementation of bypass handler
async fn bypass_handler_impl(
    provider: Option<Arc<BypassProvider>>,
    classifier: Arc<PathClassifier>,
    req: Request,
) -> Result<Response, BypassError> {
    let path = req.uri().path().to_string(); // Clone path before moving req
    let method = req.method().clone();

    // Check if we should bypass this path
    if !classifier.should_bypass(&path) {
        // Path is intercepted but router didn't match (404)
        debug!("Path {} is intercepted but no handler matched", path);
        return Ok((StatusCode::NOT_FOUND, "Not Found").into_response());
    }

    // Get provider
    let provider = match provider {
        Some(p) => p,
        None => {
            warn!(
                "Bypass enabled but no provider configured for path: {}",
                path
            );
            return Err(BypassError::NoProviderAvailable);
        }
    };

    debug!(
        "Bypassing path {} to provider {} (bypass enabled)",
        path, provider.name
    );

    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or(&path)
        .to_string();

    // Extract headers and body
    let (parts, body) = req.into_parts();
    let headers = parts.headers;

    // Read body
    let body_bytes = match axum::body::to_bytes(body, provider.max_request_body_bytes).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return Err(BypassError::BodyReadError(e.to_string()));
        }
    };

    // Proxy the request
    proxy_request(&provider, &path_and_query, method, headers, body_bytes).await
}

/// Direct proxy a request to the bypass provider
///
/// # Arguments
/// * `provider` - The bypass provider to send the request to
/// * `path` - The API path (e.g., "/v1/embeddings")
/// * `method` - HTTP method
/// * `headers` - Request headers from the client
/// * `body` - Request body bytes
///
/// # Returns
/// Returns the raw response from the provider, preserving status, headers, and body
pub async fn proxy_request(
    provider: &BypassProvider,
    path: &str,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, BypassError> {
    let url = provider.build_url(path);

    debug!(
        "Bypass proxy: {} {} -> {} (provider: {})",
        method, path, url, provider.name
    );

    let mut request_headers = forwardable_headers(&headers);
    request_headers.remove("host");

    // Client credentials determine the account and billing mode. Server keys are fallbacks.
    if !has_client_auth(&request_headers) && !provider.api_key.is_empty() {
        if provider.name == "anthropic" || provider.base_url.contains("anthropic.com") {
            request_headers.insert(
                "x-api-key",
                provider
                    .api_key
                    .parse()
                    .map_err(|_| BypassError::InvalidHeaderValue("Configured API key".into()))?,
            );
        } else {
            request_headers.insert(
                "authorization",
                format!("Bearer {}", provider.api_key)
                    .parse()
                    .map_err(|_| BypassError::InvalidHeaderValue("Configured API key".into()))?,
            );
        }
    }

    // Build reqwest request
    let mut req_builder = provider
        .client
        .request(method.clone(), &url)
        .body(body.to_vec());

    req_builder = req_builder.headers(request_headers);

    // Send request
    let response = req_builder.send().await?;

    // Extract status and headers from provider response
    let status = response.status();
    let provider_headers = response.headers().clone();

    // Stream the upstream response back to the client without buffering it.
    // Eliminates the response-side memory-exhaustion DoS: the proxy never
    // holds the full response body.
    let body_stream = response.bytes_stream();

    debug!("Bypass proxy streaming response back, status: {}", status);

    // Build axum response with a streaming body
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status;
    *response.headers_mut() = forwardable_headers(&provider_headers);

    Ok(response)
}

fn has_client_auth(headers: &HeaderMap) -> bool {
    ["authorization", "x-api-key"].iter().any(|name| {
        headers
            .get_all(*name)
            .iter()
            .any(|value| !value.as_bytes().is_empty())
    })
}

/// Preserve repeated end-to-end headers while rebuilding HTTP framing.
pub(crate) fn forwardable_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = headers.clone();
    for value in headers.get_all("connection") {
        if let Ok(value) = value.to_str() {
            for name in value.split(',') {
                if let Ok(name) = axum::http::HeaderName::from_bytes(name.trim().as_bytes()) {
                    forwarded.remove(name);
                }
            }
        }
    }
    let names: Vec<_> = forwarded
        .keys()
        .filter(|name| {
            is_hop_by_hop_header(name.as_str())
                || matches!(
                    name.as_str(),
                    "content-length" | "trailer" | "proxy-connection"
                )
        })
        .cloned()
        .collect();
    for name in names {
        forwarded.remove(name);
    }
    forwarded
}

/// Check if a header is a hop-by-hop header that should not be forwarded
fn is_hop_by_hop_header(name: &str) -> bool {
    matches!(
        name.to_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bypass_error_with_quote_in_display_produces_valid_json() {
        // A BypassError whose Display contains a double-quote and backslash
        // must serialize to valid JSON, not a broken literal.
        use axum::body::to_bytes;
        use axum::http::StatusCode;

        // RequestFailed wraps a reqwest::Error; constructing one directly is
        // awkward, so use BodyReadError which carries an arbitrary string.
        let err = BypassError::BodyReadError(r#"he said "hi" \n done"#.to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes)
            .expect("error body must be valid JSON even with quotes/backslashes");
        assert_eq!(parsed["error"], "bypass_proxy_error");
        assert_eq!(
            parsed["message"],
            r#"Body read error: he said "hi" \n done"#
        );
    }

    #[test]
    fn test_is_hop_by_hop_header() {
        assert!(is_hop_by_hop_header("Connection"));
        assert!(is_hop_by_hop_header("connection"));
        assert!(is_hop_by_hop_header("Keep-Alive"));
        assert!(is_hop_by_hop_header("Transfer-Encoding"));
        assert!(is_hop_by_hop_header("Upgrade"));

        assert!(!is_hop_by_hop_header("Content-Type"));
        assert!(!is_hop_by_hop_header("Authorization"));
        assert!(!is_hop_by_hop_header("User-Agent"));
    }

    #[test]
    fn test_bypass_provider_build_url() {
        let client = Arc::new(Client::new());
        // Bypass provider should use domain-only base URL (without /v1)
        // since request paths already include the full path
        let provider = BypassProvider::new(
            "https://api.openai.com".to_string(),
            "test-key".to_string(),
            "openai".to_string(),
            client,
            usize::MAX,
        );

        assert_eq!(
            provider.build_url("/v1/embeddings"),
            "https://api.openai.com/v1/embeddings"
        );

        assert_eq!(
            provider.build_url("v1/embeddings"),
            "https://api.openai.com/v1/embeddings"
        );
    }

    #[test]
    fn test_bypass_provider_build_url_anthropic() {
        let client = Arc::new(Client::new());
        let provider = BypassProvider::new(
            "https://api.anthropic.com".to_string(),
            "test-key".to_string(),
            "anthropic".to_string(),
            client,
            usize::MAX,
        );

        assert_eq!(
            provider.build_url("/v1/messages/count_tokens"),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
    }

    #[test]
    fn test_bypass_provider_base_url_trimming() {
        let client = Arc::new(Client::new());

        // With trailing slash
        let provider1 = BypassProvider::new(
            "https://api.openai.com/".to_string(),
            "key".to_string(),
            "test".to_string(),
            client.clone(),
            usize::MAX,
        );
        assert_eq!(provider1.base_url, "https://api.openai.com");

        // Without trailing slash
        let provider2 = BypassProvider::new(
            "https://api.openai.com".to_string(),
            "key".to_string(),
            "test".to_string(),
            client.clone(),
            usize::MAX,
        );
        assert_eq!(provider2.base_url, "https://api.openai.com");
    }
}
