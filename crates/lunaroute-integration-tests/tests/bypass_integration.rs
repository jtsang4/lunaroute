//! Integration tests for the bypass feature
//!
//! Tests that unknown API paths are bypassed directly to the provider
//! without going through the routing engine.

use axum::{
    Router,
    body::Body,
    extract::Request,
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use lunaroute_ingress::{BypassProvider, with_bypass};
use lunaroute_routing::PathClassifier;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceExt;

/// Mock provider server for testing
async fn mock_provider_server() -> String {
    let app = Router::new()
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/audio/transcriptions", post(handle_audio))
        .route("/v1/images/generations", post(handle_images))
        .route("/v1/chat/completions", post(handle_chat_completions));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    format!("http://127.0.0.1:{}", addr.port())
}

async fn handle_embeddings() -> Response {
    (
        StatusCode::OK,
        r#"{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2,0.3],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":5,"total_tokens":5}}"#,
    )
        .into_response()
}

async fn handle_audio() -> Response {
    (
        StatusCode::OK,
        r#"{"text":"Hello, this is a transcription test."}"#,
    )
        .into_response()
}

async fn handle_images() -> Response {
    (
        StatusCode::OK,
        r#"{"created":1234567890,"data":[{"url":"https://example.com/image.png"}]}"#,
    )
        .into_response()
}

async fn handle_chat_completions() -> Response {
    (
        StatusCode::OK,
        r#"{"id":"chatcmpl-123","object":"chat.completion","created":1234567890,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
    )
        .into_response()
}

#[tokio::test]
async fn test_bypass_enabled_proxies_embeddings() {
    // Start mock provider
    let provider_url = mock_provider_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create bypass provider
    let bypass_provider = Arc::new(BypassProvider::new(
        provider_url.clone(),
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        usize::MAX,
    ));

    // Create path classifier with bypass enabled
    let classifier = Arc::new(PathClassifier::new(true));

    // Create a simple router with no routes (all requests will go to fallback)
    let app = Router::new();

    // Wrap with bypass
    let app = with_bypass(app, Some(bypass_provider), classifier);

    // Test embeddings endpoint (should be bypassed)
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"text-embedding-3-small","input":"test"}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(body_str.contains("embedding"));
}

#[tokio::test]
async fn test_bypass_enabled_proxies_audio() {
    // Start mock provider
    let provider_url = mock_provider_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create bypass provider
    let bypass_provider = Arc::new(BypassProvider::new(
        provider_url,
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        usize::MAX,
    ));

    // Create path classifier with bypass enabled
    let classifier = Arc::new(PathClassifier::new(true));

    // Create a simple router
    let app = Router::new();
    let app = with_bypass(app, Some(bypass_provider), classifier);

    // Test audio endpoint (should be bypassed)
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/audio/transcriptions")
        .header("content-type", "multipart/form-data")
        .body(Body::from(r#"{"file":"audio.mp3"}"#))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(body_str.contains("transcription"));
}

#[tokio::test]
async fn test_bypass_enabled_proxies_images() {
    // Start mock provider
    let provider_url = mock_provider_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create bypass provider
    let bypass_provider = Arc::new(BypassProvider::new(
        provider_url,
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        usize::MAX,
    ));

    // Create path classifier with bypass enabled
    let classifier = Arc::new(PathClassifier::new(true));

    // Create a simple router
    let app = Router::new();
    let app = with_bypass(app, Some(bypass_provider), classifier);

    // Test images endpoint (should be bypassed)
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"prompt":"a cat","n":1}"#))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(body_str.contains("url"));
}

#[tokio::test]
async fn test_bypass_rejects_oversized_request_body() {
    let provider_url = mock_provider_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Small limit: 16 bytes.
    let bypass_provider = Arc::new(BypassProvider::new(
        provider_url,
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        16,
    ));
    let classifier = Arc::new(PathClassifier::new(true));
    let app = Router::new();
    let app = with_bypass(app, Some(bypass_provider), classifier);

    // Body of 64 bytes (well over the 16-byte limit). The bypass handler's
    // own to_bytes bound rejects it before forwarding.
    let big_body = "x".repeat(64);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(big_body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_GATEWAY,
        "oversized body must be rejected before forwarding"
    );
}

#[tokio::test]
async fn test_bypass_streams_response_body_intact() {
    // Regression: after streaming the upstream response back via
    // Body::from_stream, the client must still receive the full body.
    let provider_url = mock_provider_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let bypass_provider = Arc::new(BypassProvider::new(
        provider_url,
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        usize::MAX,
    ));
    let classifier = Arc::new(PathClassifier::new(true));
    let app = Router::new();
    let app = with_bypass(app, Some(bypass_provider), classifier);

    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"text-embedding-3-small","input":"test"}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Collect the streamed body on the client side.
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        body_str.contains("embedding"),
        "streamed response body must contain the upstream payload, got: {body_str}"
    );
}

#[tokio::test]
async fn test_bypass_disabled_returns_404() {
    // Create path classifier with bypass DISABLED
    let classifier = Arc::new(PathClassifier::new(false));

    // Create a simple router with no routes
    let app = Router::new();
    let app = with_bypass(app, None, classifier);

    // Test embeddings endpoint (should return 404 since bypass is disabled)
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"test","input":"test"}"#))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    // Should return 404 because bypass is disabled and no route matches
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_intercepted_path_not_bypassed() {
    // Start mock provider
    let provider_url = mock_provider_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create bypass provider
    let bypass_provider = Arc::new(BypassProvider::new(
        provider_url,
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        usize::MAX,
    ));

    // Create path classifier with bypass enabled
    let classifier = Arc::new(PathClassifier::new(true));

    // Create a router with a handler for chat/completions (intercepted path)
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { (StatusCode::OK, "from router handler") }),
    );

    let app = with_bypass(app, Some(bypass_provider), classifier);

    // Test chat/completions endpoint (should go through router, not bypass)
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"gpt-4","messages":[]}"#))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // Should get response from router handler, not bypassed to provider
    assert_eq!(body_str, "from router handler");
}

#[tokio::test]
async fn test_unknown_intercepted_path_returns_404() {
    // Create path classifier with bypass enabled
    let classifier = Arc::new(PathClassifier::new(true));

    // Create bypass provider
    let bypass_provider = Arc::new(BypassProvider::new(
        "http://localhost:9999".to_string(),
        "test-key".to_string(),
        "test-provider".to_string(),
        Arc::new(reqwest::Client::new()),
        usize::MAX,
    ));

    // Create a router with no routes
    let app = Router::new();
    let app = with_bypass(app, Some(bypass_provider), classifier);

    // Test /healthz (intercepted path but no handler)
    let request = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    // Should return 404 because it's an intercepted path but no handler exists
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

async fn automatic_proxy(
    openai: Option<&wiremock::MockServer>,
    anthropic: Option<&wiremock::MockServer>,
) -> Router {
    use lunaroute_egress::{
        anthropic::{AnthropicConfig, AnthropicConnector},
        openai::{OpenAIConfig, OpenAIConnector},
    };
    let openai_connector = if let Some(server) = openai {
        Some(Arc::new(
            OpenAIConnector::new(
                OpenAIConfig::new("server-openai-key")
                    .with_base_url(format!("{}/v1", server.uri())),
            )
            .await
            .unwrap(),
        ))
    } else {
        None
    };
    let anthropic_connector = anthropic.map(|server| {
        Arc::new(
            AnthropicConnector::new(
                AnthropicConfig::new("server-anthropic-key").with_base_url(server.uri()),
            )
            .unwrap(),
        )
    });
    let app = lunaroute_ingress::multi_dialect::passthrough_router(
        openai_connector,
        anthropic_connector,
        None,
        None,
        None,
        15,
        false,
        None,
    );
    let provider = |name: &str, server: &wiremock::MockServer| {
        Arc::new(BypassProvider::new(
            server.uri(),
            format!("server-{name}-key"),
            name.into(),
            Arc::new(reqwest::Client::new()),
            1024 * 1024,
        ))
    };
    lunaroute_ingress::with_auto_bypass(
        app,
        openai.map(|server| provider("openai", server)),
        anthropic.map(|server| provider("anthropic", server)),
        Arc::new(PathClassifier::new(true)),
    )
    .merge(Router::new().route("/healthz", axum::routing::get(|| async { StatusCode::OK })))
}

#[tokio::test]
async fn automatic_bypass_routes_auxiliary_requests_without_rewriting_them() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};
    let openai = MockServer::start().await;
    let anthropic = MockServer::start().await;
    for server in [&openai, &anthropic] {
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_string("raw response"))
            .mount(server)
            .await;
    }
    let app = automatic_proxy(Some(&openai), Some(&anthropic)).await;
    for (uri, header, value, body, expected) in [
        (
            "/v1/messages/count_tokens?beta=true&tag=a%2Fb&tag=a+b",
            "authorization",
            "Bearer sk-ant-oat-test",
            " {\"model\":\"claude-test\",\"extra\":true} ",
            &anthropic,
        ),
        (
            "/v1/embeddings",
            "authorization",
            "Bearer sk-openai-test",
            "{\"model\":\"text-embedding-3-small\",\"input\":\"hi\"}",
            &openai,
        ),
        (
            "/v1/audio/transcriptions",
            "authorization",
            "Bearer sk-openai-test",
            "--multipart\r\nraw bytes\r\n--multipart--",
            &openai,
        ),
        (
            "/v1/models?limit=1000",
            "anthropic-version",
            "2026-10-21",
            "",
            &anthropic,
        ),
        (
            "/v1/models?limit=10",
            "authorization",
            "Bearer sk-openai-test",
            "",
            &openai,
        ),
        (
            "/v1/files",
            "anthropic-beta",
            "files-api-2025-04-14",
            "raw upload",
            &anthropic,
        ),
        (
            "/v1/files",
            "openai-project",
            "proj_test",
            "raw upload",
            &openai,
        ),
        (
            "/v1/future",
            "content-type",
            "application/json",
            " {\"model\":\"claude-future\",\"unknown\":{\"a\":1}} ",
            &anthropic,
        ),
        (
            "/v1/future",
            "content-type",
            "application/json",
            "{\"model\":\"gpt-future\",\"unknown\":true}",
            &openai,
        ),
    ] {
        let before = expected.received_requests().await.unwrap().len();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(if uri.starts_with("/v1/models") {
                        Method::GET
                    } else {
                        Method::POST
                    })
                    .uri(uri)
                    .header(header, value)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
        let requests = expected.received_requests().await.unwrap();
        assert_eq!(requests.len(), before + 1, "{uri}");
        let captured = requests.last().unwrap();
        assert_eq!(captured.url.path(), uri.split('?').next().unwrap());
        assert_eq!(
            captured.url.query(),
            uri.split_once('?').map(|(_, query)| query)
        );
        assert_eq!(captured.body, body.as_bytes());
        assert_eq!(captured.headers[header], value);
    }
}

#[tokio::test]
async fn automatic_proxy_preserves_native_auth_and_anthropic_capabilities() {
    use serde_json::json;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};
    let openai = MockServer::start().await;
    let anthropic = MockServer::start().await;
    for server in [&openai, &anthropic] {
        Mock::given(any())
            .respond_with(|request: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                if body["stream"] == true {
                    ResponseTemplate::new(200).insert_header("content-type", "text/event-stream")
                        .set_body_string("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-test\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
                } else {
                    ResponseTemplate::new(200).set_body_json(json!({"unchanged":true}))
                }
            })
            .mount(server)
            .await;
    }
    for (enable_openai, stream) in [(true, false), (false, false), (true, true), (false, true)] {
        let app = automatic_proxy(enable_openai.then_some(&openai), Some(&anthropic)).await;
        let body = json!({"model":"claude-test","max_tokens":10,"stream":stream,
            "messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"adaptive"},"output_config":{"effort":"high"},
            "future_field":{"preserve":true}});
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/messages?beta=true")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer sk-ant-oat-client")
                    .header("anthropic-version", "2026-10-21")
                    .header("anthropic-beta", "oauth-2025-04-20")
                    .header("anthropic-beta", "context-management-2025-06-27")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        if stream {
            assert!(
                String::from_utf8(bytes.to_vec())
                    .unwrap()
                    .contains("message_stop")
            );
        }
        let requests = anthropic.received_requests().await.unwrap();
        let captured = requests.last().unwrap();
        assert_eq!(
            captured.headers["authorization"],
            "Bearer sk-ant-oat-client"
        );
        assert!(!captured.headers.contains_key("x-api-key"));
        assert_eq!(captured.headers["anthropic-version"], "2026-10-21");
        assert_eq!(captured.headers.get_all("anthropic-beta").iter().count(), 2);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&captured.body).unwrap(),
            body
        );
    }
    let app = automatic_proxy(Some(&openai), Some(&anthropic)).await;
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-client-openai")
                .body(Body::from(
                    r#"{"model":"gpt-test","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = openai.received_requests().await.unwrap();
    assert_eq!(
        requests.last().unwrap().headers["authorization"],
        "Bearer sk-client-openai"
    );
}

#[tokio::test]
async fn automatic_bypass_rejects_ambiguous_conflicting_and_disabled_destinations() {
    use wiremock::MockServer;
    let openai = MockServer::start().await;
    let anthropic = MockServer::start().await;
    let app = automatic_proxy(Some(&openai), Some(&anthropic)).await;
    for (uri, headers) in [
        ("/v1/files", vec![]),
        ("/v1/models", vec![("authorization", "Bearer opaque-token")]),
        (
            "/v1/embeddings",
            vec![("authorization", "Bearer sk-ant-oat-test")],
        ),
        (
            "/v1/chat/completions",
            vec![("authorization", "Bearer sk-ant-oat-test")],
        ),
        (
            "/v1/messages",
            vec![("authorization", "Bearer sk-openai-test")],
        ),
        (
            "/v1/messages/count_tokens",
            vec![("authorization", "Bearer sk-openai-test")],
        ),
        (
            "/v1/files",
            vec![
                ("anthropic-version", "2023-06-01"),
                ("openai-project", "proj_test"),
            ],
        ),
    ] {
        let mut request = Request::builder().method(Method::GET).uri(uri);
        for (header, value) in headers {
            request = request.header(header, value);
        }
        let response = app
            .clone()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
    }
    let app = automatic_proxy(Some(&openai), None).await;
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("authorization", "Bearer sk-ant-oat-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(openai.received_requests().await.unwrap().is_empty());
    assert!(anthropic.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn automatic_bypass_preserves_repeated_headers_raw_streams_and_http_errors() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};
    for (status, content_type, body) in [
        (
            200,
            "text/event-stream",
            ": heartbeat\nevent: future_event\nid: 1\nretry: 1000\ndata: {\"unknown\":true}\n\n",
        ),
        (
            429,
            "application/json",
            " {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"limit\"}} ",
        ),
    ] {
        let openai = MockServer::start().await;
        let anthropic = MockServer::start().await;
        Mock::given(any())
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("content-type", content_type)
                    .insert_header("request-id", "req_test")
                    .insert_header("retry-after", "120")
                    .insert_header("x-should-retry", "false")
                    .insert_header("anthropic-ratelimit-unified-status", "allowed")
                    .append_header("set-cookie", "first=1")
                    .append_header("set-cookie", "second=2")
                    .insert_header("connection", "x-private-response")
                    .insert_header("x-private-response", "secret")
                    .set_body_string(body),
            )
            .mount(&anthropic)
            .await;
        let app = automatic_proxy(Some(&openai), Some(&anthropic)).await;
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/messages/count_tokens?beta=true")
                    .header("authorization", "Bearer sk-ant-oat-client")
                    .header("anthropic-beta", "oauth-2025-04-20")
                    .header("anthropic-beta", "context-management-2025-06-27")
                    .header("connection", "x-private-request")
                    .header("x-private-request", "secret")
                    .body(Body::from("raw body"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), status);
        assert_eq!(response.headers()["request-id"], "req_test");
        assert_eq!(response.headers()["retry-after"], "120");
        assert_eq!(response.headers()["x-should-retry"], "false");
        assert_eq!(
            response.headers()["anthropic-ratelimit-unified-status"],
            "allowed"
        );
        assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
        assert!(!response.headers().contains_key("x-private-response"));
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(bytes, body.as_bytes());
        let requests = anthropic.received_requests().await.unwrap();
        assert_eq!(
            requests.len(),
            1,
            "the proxy must leave retries to the client"
        );
        assert_eq!(
            requests[0].headers.get_all("anthropic-beta").iter().count(),
            2
        );
        assert!(!requests[0].headers.contains_key("x-private-request"));
        assert!(!requests[0].headers.contains_key("x-api-key"));
        assert!(openai.received_requests().await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn automatic_bypass_keeps_client_keys_and_falls_back_to_server_keys() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};
    let openai = MockServer::start().await;
    let anthropic = MockServer::start().await;
    for server in [&openai, &anthropic] {
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(server)
            .await;
    }
    let app = automatic_proxy(Some(&openai), Some(&anthropic)).await;
    for (uri, client_auth, header, expected, server) in [
        (
            "/v1/messages/count_tokens",
            None,
            "x-api-key",
            "server-anthropic-key",
            &anthropic,
        ),
        (
            "/v1/messages/count_tokens",
            Some(("x-api-key", "sk-ant-api-client")),
            "x-api-key",
            "sk-ant-api-client",
            &anthropic,
        ),
        (
            "/v1/embeddings",
            None,
            "authorization",
            "Bearer server-openai-key",
            &openai,
        ),
        (
            "/v1/embeddings",
            Some(("authorization", "Bearer sk-client-openai")),
            "authorization",
            "Bearer sk-client-openai",
            &openai,
        ),
    ] {
        let mut request = Request::builder().method(Method::POST).uri(uri);
        if let Some((name, value)) = client_auth {
            request = request.header(name, value);
        }
        let response = app
            .clone()
            .oneshot(request.body(Body::from("{}")).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.last().unwrap().headers[header], expected);
    }
    let app = automatic_proxy(None, Some(&anthropic)).await;
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = anthropic.received_requests().await.unwrap();
    assert_eq!(requests.last().unwrap().url.path(), "/v1/models");
    assert_eq!(
        requests.last().unwrap().headers["x-api-key"],
        "server-anthropic-key"
    );
}
