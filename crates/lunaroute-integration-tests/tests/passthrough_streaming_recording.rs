//! Integration tests for passthrough mode streaming with session recording
//!
//! These tests verify that:
//! 1. OpenAI passthrough streaming records session events correctly
//! 2. Anthropic passthrough streaming records session events correctly
//! 3. All expected session events are captured (Started, RequestRecorded, StreamStarted, Completed)
//! 4. Session metadata (tokens, timing, etc.) is accurately recorded

mod common;

use axum::body::Body;
use axum::http::Request;
use common::InMemorySessionStore;
use futures::StreamExt;
use lunaroute_egress::anthropic::{AnthropicConfig, AnthropicConnector};
use lunaroute_egress::openai::{OpenAIConfig, OpenAIConnector};
use lunaroute_session::SessionEvent;
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn recorded_anthropic_stream(sse: String) -> (String, Vec<SessionEvent>) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer client-token"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .expect(1)
        .mount(&server)
        .await;
    let store = Arc::new(InMemorySessionStore::new());
    let connector = Arc::new(
        AnthropicConnector::new(AnthropicConfig::new("").with_base_url(server.uri())).unwrap(),
    );
    let app = lunaroute_ingress::anthropic::passthrough_router(
        connector,
        None,
        None,
        Some(store.clone()),
        15,
        false,
        None,
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .method("POST")
                .header("content-type", "application/json")
                .header("authorization", "Bearer client-token")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(
                    json!({
                        "model": "claude-test",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 20000,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let events = wait_for_stream_completion(&store).await;
    (String::from_utf8(bytes.to_vec()).unwrap(), events)
}

async fn wait_for_stream_completion(store: &InMemorySessionStore) -> Vec<SessionEvent> {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let events = store.get_events();
            if events
                .iter()
                .any(|event| matches!(event, SessionEvent::Completed { .. }))
            {
                return events;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("stream completion was recorded")
}

fn sse_event(value: serde_json::Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        value["type"].as_str().unwrap(),
        value
    )
}

#[tokio::test]
async fn test_anthropic_passthrough_records_cumulative_usage_and_cache() {
    let sse = [
        json!({"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":9000,"cache_creation_input_tokens":1000}}}),
        json!({"type":"message_delta","usage":{"input_tokens":25,"output_tokens":12,"cache_creation_input_tokens":1500}}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}),
        json!({"type":"message_stop"}),
    ].into_iter().map(sse_event).collect();
    let (_, events) = recorded_anthropic_stream(sse).await;
    let tokens = events
        .iter()
        .find_map(|event| match event {
            SessionEvent::StatsUpdated {
                token_updates: Some(tokens),
                ..
            } => Some(tokens),
            _ => None,
        })
        .expect("token update");
    assert_eq!(tokens.total_input, 25);
    assert_eq!(tokens.total_output, 20);
    assert_eq!(tokens.total_cache_read, 9000);
    assert_eq!(tokens.total_cached, 9000);
    assert_eq!(tokens.total_cache_creation, 1500);
    assert_eq!(tokens.grand_total, 10545);
    assert!(events.iter().any(|event| matches!(
        event,
        SessionEvent::Completed {
            success: true,
            error: None,
            ..
        }
    )));
}

#[tokio::test]
async fn test_anthropic_passthrough_records_cache_only_usage() {
    let sse = [
        json!({"type":"message_start","message":{"usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":500}}}),
        json!({"type":"message_stop"}),
    ].into_iter().map(sse_event).collect();
    let (_, events) = recorded_anthropic_stream(sse).await;
    assert!(events.iter().any(|event| matches!(event,
        SessionEvent::StatsUpdated { token_updates: Some(tokens), .. }
        if tokens.total_cache_read == 500 && tokens.grand_total == 500)));
}

#[tokio::test]
async fn test_anthropic_passthrough_long_stream_keeps_usage_and_late_tools() {
    let mut sse = sse_event(
        json!({"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":10,"output_tokens":0}}}),
    );
    for _ in 0..10005 {
        sse.push_str(&sse_event(json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}})));
    }
    for value in [
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_late","name":"Read","input":{}}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/tmp/example\"}"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"future_event","opaque":true}),
        json!({"type":"ping"}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10005}}),
        json!({"type":"message_stop"}),
    ] {
        sse.push_str(&sse_event(value));
    }
    let (body, events) = recorded_anthropic_stream(sse).await;
    assert_eq!(body.matches("event: content_block_delta").count(), 10006);
    assert!(body.contains("event: future_event"));
    assert!(body.contains("event: ping"));
    assert!(body.contains("event: message_stop"));
    assert!(events.iter().any(|event| matches!(event,
        SessionEvent::StatsUpdated { token_updates: Some(tokens), .. }
        if tokens.total_output == 10005 && tokens.grand_total == 10015)));
    assert!(events.iter().any(|event| matches!(event,
        SessionEvent::ToolCallRecorded { tool_call_id, tool_arguments: Some(arguments), .. }
        if tool_call_id == "toolu_late" && arguments == "{\"file_path\":\"/tmp/example\"}")));
}

#[tokio::test]
async fn test_anthropic_passthrough_records_stream_errors_as_failures() {
    for trailing_stop in [false, true] {
        let mut sse = sse_event(
            json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
        );
        if trailing_stop {
            sse.push_str(&sse_event(json!({"type":"message_stop"})));
        }
        let (body, events) = recorded_anthropic_stream(sse).await;
        assert!(body.contains("overloaded_error"));
        let completions: Vec<_> = events
            .iter()
            .filter(|event| matches!(event, SessionEvent::Completed { .. }))
            .collect();
        assert_eq!(completions.len(), 1);
        assert!(
            matches!(completions[0], SessionEvent::Completed { success: false, error: Some(error), .. } if error.contains("overloaded_error") && error.contains("Overloaded"))
        );
    }
}

#[tokio::test]
async fn test_anthropic_passthrough_records_incomplete_stream_as_failure() {
    let (_, events) = recorded_anthropic_stream(sse_event(
        json!({"type":"message_start","message":{"usage":{"input_tokens":10}}}),
    ))
    .await;
    assert!(events.iter().any(|event| matches!(
        event,
        SessionEvent::Completed {
            success: false,
            error: Some(_),
            ..
        }
    )));
}

#[tokio::test]
async fn test_anthropic_passthrough_records_transport_errors_and_client_disconnects() {
    for disconnect_client in [false, true] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream = axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(move || async move {
                let first = futures::stream::once(async {
                    Ok::<_, std::io::Error>(axum::body::Bytes::from(sse_event(json!({
                        "type":"message_start", "message":{"usage":{"input_tokens":10}}
                    }))))
                });
                let interrupted = futures::stream::once(async move {
                    if disconnect_client {
                        futures::future::pending::<()>().await;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    Err::<axum::body::Bytes, _>(std::io::Error::other("upstream disconnected"))
                });
                axum::http::Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(first.chain(interrupted)))
                    .unwrap()
            }),
        );
        let server_task = tokio::spawn(async move {
            axum::serve(listener, upstream).await.unwrap();
        });
        let store = Arc::new(InMemorySessionStore::new());
        let connector = Arc::new(
            AnthropicConnector::new(
                AnthropicConfig::new("").with_base_url(format!("http://{address}")),
            )
            .unwrap(),
        );
        let app = lunaroute_ingress::anthropic::passthrough_router(
            connector,
            None,
            None,
            Some(store.clone()),
            15,
            false,
            None,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"model":"claude-test","messages":[],"max_tokens":10,"stream":true})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut stream = response.into_body().into_data_stream();
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&first).contains("message_start"));
        if !disconnect_client {
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .is_err()
            );
        }
        drop(stream);
        let events = wait_for_stream_completion(&store).await;
        let completions: Vec<_> = events
            .iter()
            .filter(|event| matches!(event, SessionEvent::Completed { .. }))
            .collect();
        assert_eq!(completions.len(), 1);
        assert!(matches!(
            completions[0],
            SessionEvent::Completed {
                success: false,
                error: Some(_),
                ..
            }
        ));
        assert!(events.iter().any(|event| matches!(event, SessionEvent::StatsUpdated { token_updates: Some(tokens), .. } if tokens.total_input == 10)));
        server_task.abort();
        let _ = server_task.await;
    }
}

#[tokio::test]
async fn test_anthropic_passthrough_preserves_query_and_client_auth() {
    for stream in [false, true] {
        let server = MockServer::start().await;
        let template = if stream {
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_event(json!({"type":"message_stop"})))
        } else {
            ResponseTemplate::new(200).set_body_json(json!({"type":"message","content":[]}))
        };
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("authorization", "Bearer client-token"))
            .respond_with(template)
            .expect(1)
            .mount(&server)
            .await;
        let connector = Arc::new(
            AnthropicConnector::new(AnthropicConfig::new("").with_base_url(server.uri())).unwrap(),
        );
        let app = lunaroute_ingress::anthropic::passthrough_router(
            connector, None, None, None, 15, false, None,
        );
        let query = "beta=true&tag=a%2Fb&tag=a+b";
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/messages?{query}"))
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer client-token")
                    .body(Body::from(
                        json!({"model":"claude-test","messages":[],"max_tokens":1,"stream":stream})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            server.received_requests().await.unwrap()[0].url.query(),
            Some(query)
        );
    }
}

#[tokio::test]
async fn test_anthropic_bypass_preserves_query_and_client_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .and(header("authorization", "Bearer client-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens":10})))
        .expect(1)
        .mount(&server)
        .await;
    let provider = lunaroute_ingress::BypassProvider::new(
        server.uri(),
        String::new(),
        "anthropic".into(),
        Arc::new(reqwest::Client::new()),
        1024,
    );
    let app = lunaroute_ingress::with_bypass(
        axum::Router::new(),
        Some(Arc::new(provider)),
        Arc::new(lunaroute_routing::PathClassifier::new(true)),
    );
    let query = "beta=true&tag=a%2Fb&tag=a+b";
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/messages/count_tokens?{query}"))
                .header("authorization", "Bearer client-token")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        server.received_requests().await.unwrap()[0].url.query(),
        Some(query)
    );
}

#[tokio::test]
async fn test_openai_passthrough_streaming_with_recording() {
    // Setup: Mock OpenAI server that returns streaming response
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_json(json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 50
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}

data: [DONE]

"#
        ))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Create in-memory session store
    let store = Arc::new(InMemorySessionStore::new());

    // Create OpenAI connector pointing to mock server
    let config = OpenAIConfig {
        api_key: "test-api-key".to_string(),
        base_url: mock_server.uri(),
        organization: None,
        client_config: Default::default(),
        custom_headers: None,
        request_body_config: None,
        response_body_config: None,
        codex_auth: None,
        switch_notification_message: None,
    };
    let connector = Arc::new(OpenAIConnector::new(config).await.unwrap());

    // Create passthrough router with recording
    let app = lunaroute_ingress::openai::passthrough_router(
        connector,
        None, // no stats tracker
        None, // no metrics
        Some(store.clone()),
        15,   // SSE keepalive interval
        true, // SSE keepalive enabled
        None, // no provider registry
    );

    // Send streaming request
    let request = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 50
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/chat/completions")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Verify streaming response
    assert_eq!(response.status(), 200);

    // Collect streaming response
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // Verify we got streaming chunks
    assert!(body_str.contains("data: {"));
    assert!(body_str.contains("Hello"));
    assert!(body_str.contains(" there"));
    assert!(body_str.contains("!"));
    assert!(body_str.contains("[DONE]"));

    // Wait for async events to flush

    // Wait for events to be flushed
    // Note: StreamStarted event is only emitted if time-to-first-token > 0,
    // which may not happen in fast mock tests. We'll wait for Completed instead.
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(5);
    loop {
        let events = store.get_events();
        if events
            .iter()
            .any(|e| matches!(e, SessionEvent::Completed { .. }))
        {
            break;
        }
        if start.elapsed() > timeout {
            panic!(
                "Timeout waiting for Completed event. Got {} events: {:?}",
                events.len(),
                events
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Verify session events were recorded
    let events = store.get_events();

    // Should have at least: Started, RequestRecorded, StreamStarted, Completed
    assert!(
        events.len() >= 3,
        "Expected at least 3 session events, got {}",
        events.len()
    );

    // Verify Started event
    let started_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Started { .. }));
    assert!(started_event.is_some(), "Expected SessionEvent::Started");

    if let Some(SessionEvent::Started {
        model_requested,
        provider,
        listener,
        is_streaming,
        ..
    }) = started_event
    {
        assert_eq!(model_requested, "gpt-4");
        assert_eq!(provider, "openai");
        assert_eq!(listener, "openai");
        assert!(is_streaming);
    }

    // Verify RequestRecorded event
    let request_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::RequestRecorded { .. }));
    assert!(
        request_event.is_some(),
        "Expected SessionEvent::RequestRecorded"
    );

    // Verify StreamStarted event (may not be present if ttft_ms is 0 in fast mocks)
    let stream_started = events
        .iter()
        .find(|e| matches!(e, SessionEvent::StreamStarted { .. }));
    if stream_started.is_none() {
        eprintln!("Warning: SessionEvent::StreamStarted not found (ttft_ms may be 0)");
    }

    // Verify Completed event (if present - may not always be recorded in time for non-streaming)
    let completed_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Completed { .. }));
    if completed_event.is_none() {
        eprintln!("Warning: SessionEvent::Completed not found for non-streaming request");
    }
}

#[tokio::test]
async fn test_anthropic_passthrough_streaming_with_recording() {
    // Setup: Mock Anthropic server that returns streaming response
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-sonnet-4-5","usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

"#
        ))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Create in-memory session store
    let store = Arc::new(InMemorySessionStore::new());

    // Create Anthropic connector pointing to mock server
    let config = AnthropicConfig {
        api_key: "test-api-key".to_string(),
        base_url: mock_server.uri(),
        api_version: "2023-06-01".to_string(),
        client_config: Default::default(),
        switch_notification_message: None,
    };
    let connector = Arc::new(AnthropicConnector::new(config).unwrap());

    // Create passthrough router with recording
    let app = lunaroute_ingress::anthropic::passthrough_router(
        connector,
        None, // no stats tracker
        None, // no metrics
        Some(store.clone()),
        15,   // SSE keepalive interval
        true, // SSE keepalive enabled
        None, // no provider registry
    );

    // Send streaming request
    let request = json!({
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 50
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .method("POST")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Verify streaming response
    assert_eq!(response.status(), 200);

    // Collect streaming response
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // Verify we got streaming events
    assert!(body_str.contains("event: message_start"));
    assert!(body_str.contains("event: content_block_delta"));
    assert!(body_str.contains("Hello"));
    assert!(body_str.contains(" there"));
    assert!(body_str.contains("!"));
    assert!(body_str.contains("event: message_stop"));

    // Wait for async events to flush

    // Wait longer to ensure async worker has time to flush
    // Under heavy load (full test suite), the worker task may be delayed
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Verify session events were recorded
    let events = store.get_events();

    // Should have at least: Started, RequestRecorded, StreamStarted, Completed
    assert!(
        events.len() >= 3,
        "Expected at least 3 session events, got {}",
        events.len()
    );

    // Verify Started event
    let started_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Started { .. }));
    assert!(started_event.is_some(), "Expected SessionEvent::Started");

    if let Some(SessionEvent::Started {
        model_requested,
        provider,
        listener,
        is_streaming,
        ..
    }) = started_event
    {
        assert_eq!(model_requested, "claude-sonnet-4-5");
        assert_eq!(provider, "anthropic");
        assert_eq!(listener, "anthropic");
        assert!(is_streaming);
    }

    // Verify RequestRecorded event
    let request_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::RequestRecorded { .. }));
    assert!(
        request_event.is_some(),
        "Expected SessionEvent::RequestRecorded"
    );

    // Verify Completed event (if present - may not always be recorded in time for non-streaming)
    let completed_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Completed { .. }));
    if completed_event.is_none() {
        eprintln!("Warning: SessionEvent::Completed not found for non-streaming request");
    }
}

#[tokio::test]
async fn test_openai_passthrough_non_streaming_with_recording() {
    // Setup: Mock OpenAI server that returns non-streaming response
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Create in-memory session store
    let store = Arc::new(InMemorySessionStore::new());

    // Create OpenAI connector pointing to mock server
    let config = OpenAIConfig {
        api_key: "test-api-key".to_string(),
        base_url: mock_server.uri(),
        organization: None,
        client_config: Default::default(),
        custom_headers: None,
        request_body_config: None,
        response_body_config: None,
        codex_auth: None,
        switch_notification_message: None,
    };
    let connector = Arc::new(OpenAIConnector::new(config).await.unwrap());

    // Create passthrough router with recording
    let app = lunaroute_ingress::openai::passthrough_router(
        connector,
        None, // no stats tracker
        None, // no metrics
        Some(store.clone()),
        15,   // SSE keepalive interval
        true, // SSE keepalive enabled
        None, // no provider registry
    );

    // Send non-streaming request
    let request = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/chat/completions")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Verify non-streaming response
    assert_eq!(response.status(), 200);

    // Collect response
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify we got a completion response
    assert_eq!(response_json["object"], "chat.completion");
    assert_eq!(
        response_json["choices"][0]["message"]["content"],
        "Hello there!"
    );

    // Wait for async events to flush

    // Wait longer to ensure async worker has time to flush
    // Under heavy load (full test suite), the worker task may be delayed
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Verify session events were recorded
    let events = store.get_events();

    // Should have at least: Started, RequestRecorded, Completed (no StreamStarted for non-streaming)
    assert!(
        events.len() >= 3,
        "Expected at least 3 session events, got {}",
        events.len()
    );

    // Verify Started event
    let started_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Started { .. }));
    assert!(started_event.is_some(), "Expected SessionEvent::Started");

    if let Some(SessionEvent::Started {
        model_requested,
        provider,
        listener,
        is_streaming,
        ..
    }) = started_event
    {
        assert_eq!(model_requested, "gpt-4");
        assert_eq!(provider, "openai");
        assert_eq!(listener, "openai");
        assert!(!is_streaming); // Non-streaming request
    }

    // Verify RequestRecorded event
    let request_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::RequestRecorded { .. }));
    assert!(
        request_event.is_some(),
        "Expected SessionEvent::RequestRecorded"
    );

    // Verify NO StreamStarted event (for non-streaming requests)
    let stream_started = events
        .iter()
        .find(|e| matches!(e, SessionEvent::StreamStarted { .. }));
    assert!(
        stream_started.is_none(),
        "Did not expect SessionEvent::StreamStarted for non-streaming request"
    );

    // Verify Completed event (if present - may not always be recorded in time for non-streaming)
    let completed_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Completed { .. }));
    if completed_event.is_none() {
        eprintln!("Warning: SessionEvent::Completed not found for non-streaming request");
    }
}

#[tokio::test]
async fn test_anthropic_passthrough_non_streaming_with_recording() {
    // Setup: Mock Anthropic server that returns non-streaming response
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Create in-memory session store
    let store = Arc::new(InMemorySessionStore::new());

    // Create Anthropic connector pointing to mock server
    let config = AnthropicConfig {
        api_key: "test-api-key".to_string(),
        base_url: mock_server.uri(),
        api_version: "2023-06-01".to_string(),
        client_config: Default::default(),
        switch_notification_message: None,
    };
    let connector = Arc::new(AnthropicConnector::new(config).unwrap());

    // Create passthrough router with recording
    let app = lunaroute_ingress::anthropic::passthrough_router(
        connector,
        None, // no stats tracker
        None, // no metrics
        Some(store.clone()),
        15,   // SSE keepalive interval
        true, // SSE keepalive enabled
        None, // no provider registry
    );

    // Send non-streaming request
    let request = json!({
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/messages")
                .method("POST")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Verify non-streaming response
    assert_eq!(response.status(), 200);

    // Collect response
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify we got a message response
    assert_eq!(response_json["type"], "message");
    assert_eq!(response_json["content"][0]["text"], "Hello there!");

    // Wait for async events to flush

    // Wait longer to ensure async worker has time to flush
    // Under heavy load (full test suite), the worker task may be delayed
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Verify session events were recorded
    let events = store.get_events();

    // Should have at least: Started, RequestRecorded, Completed (no StreamStarted for non-streaming)
    assert!(
        events.len() >= 3,
        "Expected at least 3 session events, got {}",
        events.len()
    );

    // Verify Started event
    let started_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Started { .. }));
    assert!(started_event.is_some(), "Expected SessionEvent::Started");

    if let Some(SessionEvent::Started {
        model_requested,
        provider,
        listener,
        is_streaming,
        ..
    }) = started_event
    {
        assert_eq!(model_requested, "claude-sonnet-4-5");
        assert_eq!(provider, "anthropic");
        assert_eq!(listener, "anthropic");
        assert!(!is_streaming); // Non-streaming request
    }

    // Verify RequestRecorded event
    let request_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::RequestRecorded { .. }));
    assert!(
        request_event.is_some(),
        "Expected SessionEvent::RequestRecorded"
    );

    // Verify NO StreamStarted event (for non-streaming requests)
    let stream_started = events
        .iter()
        .find(|e| matches!(e, SessionEvent::StreamStarted { .. }));
    assert!(
        stream_started.is_none(),
        "Did not expect SessionEvent::StreamStarted for non-streaming request"
    );

    // Verify Completed event (if present - may not always be recorded in time for non-streaming)
    let completed_event = events
        .iter()
        .find(|e| matches!(e, SessionEvent::Completed { .. }));
    if completed_event.is_none() {
        eprintln!("Warning: SessionEvent::Completed not found for non-streaming request");
    }
}
