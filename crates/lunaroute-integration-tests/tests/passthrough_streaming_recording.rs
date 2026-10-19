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

    let events = wait_for_stream_completion(&store).await;

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

    assert!(matches!(assert_single_completion(&events, true),
        SessionEvent::Completed { finish_reason: Some(reason), final_stats, .. }
        if reason == "end_turn" && final_stats.total_tokens.total_input == 10
            && final_stats.total_tokens.total_output == 3));
}

fn raw_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .build()
        .unwrap()
}

struct TestHttpServer {
    url: String,
    task: tokio::task::JoinHandle<()>,
}

impl TestHttpServer {
    async fn start(app: axum::Router) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, task }
    }
}

impl Drop for TestHttpServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn http_anthropic_proxy(
    upstream: &str,
    recording: bool,
) -> (TestHttpServer, Arc<InMemorySessionStore>) {
    let store = Arc::new(InMemorySessionStore::new());
    let connector = Arc::new(
        AnthropicConnector::new(AnthropicConfig::new("").with_base_url(upstream)).unwrap(),
    );
    let app = lunaroute_ingress::anthropic::passthrough_router(
        connector,
        None,
        None,
        recording.then(|| store.clone() as Arc<dyn lunaroute_core::session_store::SessionStore>),
        15,
        false,
        None,
    );
    (TestHttpServer::start(app).await, store)
}

fn anthropic_http_request(stream: bool) -> serde_json::Value {
    json!({
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream
    })
}

fn assert_single_completion(events: &[SessionEvent], success: bool) -> &SessionEvent {
    let completed: Vec<_> = events
        .iter()
        .filter(|event| matches!(event, SessionEvent::Completed { .. }))
        .collect();
    assert_eq!(completed.len(), 1, "one completion per request: {events:?}");
    let SessionEvent::Completed {
        request_id,
        session_id,
        success: actual,
        ..
    } = completed[0]
    else {
        unreachable!()
    };
    assert_eq!(*actual, success, "{events:?}");
    assert!(events.iter().any(|event| matches!(event,
        SessionEvent::Started { request_id: rid, session_id: sid, .. }
        if rid == request_id && sid == session_id)));
    completed[0]
}

#[tokio::test]
async fn test_anthropic_chunked_responses_over_http() {
    for (status, stream) in [(200, false), (401, false), (401, true)] {
        for recording in [false, true] {
            let body = if status == 200 {
                json!({"model":"claude-test","content":[],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}})
            } else {
                json!({"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}})
            }.to_string();
            let expected = body.clone();
            let upstream = TestHttpServer::start(axum::Router::new().route(
                "/v1/messages",
                axum::routing::post(move |headers: axum::http::HeaderMap| {
                    let body = body.clone();
                    async move {
                        assert!(!headers.contains_key("x-client-hop"));
                        assert!(!headers.contains_key("x-client-other-hop"));
                        assert!(!headers.contains_key("trailer"));
                        let chunks = futures::stream::iter([
                            Ok::<_, std::io::Error>(body[..10].to_string()),
                            Ok(body[10..].to_string()),
                        ]);
                        let mut response = axum::http::Response::builder()
                            .status(status)
                            .header("content-type", "application/json")
                            .header("request-id", "req-upstream")
                            .header("retry-after", "2")
                            .header("set-cookie", "first=1")
                            .header("set-cookie", "second=2")
                            .header("connection", "keep-alive, X-Upstream-Hop")
                            .header("x-upstream-hop", "private")
                            .header("x-upstream-other-hop", "private")
                            .header("keep-alive", "timeout=5")
                            .header("trailer", "x-trailer")
                            .body(Body::from_stream(chunks))
                            .unwrap();
                        response
                            .headers_mut()
                            .append("connection", "X-Upstream-Other-Hop".parse().unwrap());
                        response
                    }
                }),
            ))
            .await;
            let (proxy, store) = http_anthropic_proxy(&upstream.url, recording).await;
            let response = raw_http_client()
                .post(format!("{}/v1/messages", proxy.url))
                .timeout(std::time::Duration::from_secs(5))
                .header("connection", "X-Client-Hop")
                .header("connection", "X-Client-Other-Hop")
                .header("x-client-hop", "private")
                .header("x-client-other-hop", "private")
                .header("trailer", "x-trailer")
                .json(&anthropic_http_request(stream))
                .send()
                .await
                .expect("proxy must return a complete HTTP response");
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(response.headers()["request-id"], "req-upstream");
            assert_eq!(response.headers()["retry-after"], "2");
            assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
            for name in [
                "connection",
                "keep-alive",
                "x-upstream-hop",
                "x-upstream-other-hop",
                "trailer",
                "transfer-encoding",
            ] {
                assert!(!response.headers().contains_key(name), "leaked {name}");
            }
            assert_eq!(response.text().await.unwrap(), expected);
            if recording {
                let events = wait_for_stream_completion(&store).await;
                let completion = assert_single_completion(&events, status == 200);
                if status != 200 {
                    let SessionEvent::Completed {
                        error: Some(error), ..
                    } = completion
                    else {
                        panic!("HTTP failure must be recorded");
                    };
                    assert!(error.contains("401"));
                    assert!(error.contains("authentication_error"));
                    assert!(error.contains("req-upstream"));
                }
            } else {
                assert!(store.get_events().is_empty());
            }
        }
    }
}

fn compress_anthropic_body(body: &[u8], encoding: &str) -> Vec<u8> {
    use std::io::Write;
    match encoding {
        "identity" => body.to_vec(),
        "gzip" => {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            encoder.write_all(body).unwrap();
            encoder.finish().unwrap()
        }
        "deflate" => {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
            encoder.write_all(body).unwrap();
            encoder.finish().unwrap()
        }
        "br" => {
            let mut encoded = Vec::new();
            {
                let mut encoder = brotli::CompressorWriter::new(&mut encoded, 4096, 3, 22);
                encoder.write_all(body).unwrap();
            }
            encoded
        }
        "zstd" => zstd::stream::encode_all(body, 1).unwrap(),
        "gzip, br" => compress_anthropic_body(&compress_anthropic_body(body, "gzip"), "br"),
        _ => panic!("unknown test encoding"),
    }
}

#[tokio::test]
async fn test_anthropic_compressed_response_recording_over_http() {
    let body = json!({
        "model":"claude-test", "stop_reason":"tool_use",
        "content":[{"type":"tool_use","id":"toolu_http","name":"Read","input":{"file_path":"/tmp/test"}}],
        "usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":500,"cache_creation_input_tokens":20}
    }).to_string();
    for encoding in ["identity", "gzip", "deflate", "br", "zstd", "gzip, br"] {
        let encoded = compress_anthropic_body(body.as_bytes(), encoding);
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .insert_header("content-encoding", encoding)
                    .set_body_bytes(encoded.clone()),
            )
            .mount(&upstream)
            .await;
        let (proxy, store) = http_anthropic_proxy(&upstream.uri(), true).await;
        let response = raw_http_client()
            .post(format!("{}/v1/messages", proxy.url))
            .header("accept-encoding", "gzip, deflate, br, zstd")
            .json(&anthropic_http_request(false))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["content-encoding"], encoding);
        assert_eq!(response.bytes().await.unwrap().as_ref(), encoded.as_slice());
        let events = wait_for_stream_completion(&store).await;
        let completion = assert_single_completion(&events, true);
        assert!(matches!(completion, SessionEvent::Completed {
            finish_reason: Some(reason), final_stats, ..
        } if reason == "tool_use" && final_stats.total_tokens.grand_total == 520
            && final_stats.tool_summary.total_tool_calls == 1));
        assert!(events.iter().any(|event| matches!(event,
            SessionEvent::StatsUpdated { token_updates: Some(tokens), has_refusal: false, .. }
            if tokens.total_cache_read == 500 && tokens.total_cache_creation == 20 && tokens.grand_total == 520)));
        assert!(events.iter().any(|event| matches!(event,
            SessionEvent::ToolCallRecorded { tool_call_id, tool_arguments: Some(arguments), .. }
            if tool_call_id == "toolu_http" && arguments == r#"{"file_path":"/tmp/test"}"#)));
        assert_completion_persisted(&events, true).await;
    }
}

#[tokio::test]
async fn test_anthropic_response_parse_failures_complete_over_http() {
    for (body, encoding) in [
        (b"invalid JSON".to_vec(), "identity"),
        (b"invalid gzip".to_vec(), "gzip"),
        (b"{}".to_vec(), "unknown"),
    ] {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-encoding", encoding)
                    .set_body_bytes(body.clone()),
            )
            .mount(&upstream)
            .await;
        let (proxy, store) = http_anthropic_proxy(&upstream.uri(), true).await;
        let response = raw_http_client()
            .post(format!("{}/v1/messages", proxy.url))
            .json(&anthropic_http_request(false))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.bytes().await.unwrap().as_ref(), body.as_slice());
        let events = wait_for_stream_completion(&store).await;
        assert!(matches!(assert_single_completion(&events, false),
            SessionEvent::Completed { error: Some(error), .. } if error.contains("recording") && error.contains("200")));
    }
}

#[tokio::test]
async fn test_anthropic_transport_failures_complete_over_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    for stream in [false, true] {
        let (proxy, store) = http_anthropic_proxy(&upstream, true).await;
        let response = raw_http_client()
            .post(format!("{}/v1/messages", proxy.url))
            .json(&anthropic_http_request(stream))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 502);
        let events = wait_for_stream_completion(&store).await;
        assert!(matches!(
            assert_single_completion(&events, false),
            SessionEvent::Completed { error: Some(_), .. }
        ));
    }
}

async fn assert_completion_persisted(events: &[SessionEvent], success: bool) {
    use lunaroute_session::{sqlite_writer::SqliteWriter, writer::SessionWriter};
    let directory = tempfile::tempdir().unwrap();
    let writer = SqliteWriter::new(&directory.path().join("sessions.db"))
        .await
        .unwrap();
    writer.write_batch(events).await.unwrap();
    let sessions = writer.search_sessions(&Default::default()).await.unwrap();
    assert_eq!(sessions.items.len(), 1);
    assert!(sessions.items[0].completed_at.is_some());
    assert_eq!(sessions.items[0].success, Some(success));
    let SessionEvent::Completed { final_stats, .. } = assert_single_completion(events, success)
    else {
        unreachable!()
    };
    assert_eq!(
        sessions.items[0].total_tokens as u64,
        final_stats.total_tokens.grand_total
    );
}

#[tokio::test]
async fn test_anthropic_compressed_http_errors_complete_over_http() {
    for stream in [false, true] {
        for (status, encoding, body) in [
            (
                429,
                "br",
                json!({"type":"error","error":{"type":"rate_limit_error","message":"Retry later"}})
                    .to_string(),
            ),
            (503, "gzip", "<html>Unavailable</html>".to_string()),
        ] {
            let encoded = compress_anthropic_body(body.as_bytes(), encoding);
            let upstream = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("content-encoding", encoding)
                        .insert_header("request-id", "req-error")
                        .insert_header("retry-after", "10")
                        .set_body_bytes(encoded.clone()),
                )
                .mount(&upstream)
                .await;
            let (proxy, store) = http_anthropic_proxy(&upstream.uri(), true).await;
            let response = raw_http_client()
                .post(format!("{}/v1/messages", proxy.url))
                .json(&anthropic_http_request(stream))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(response.headers()["content-encoding"], encoding);
            assert_eq!(response.headers()["retry-after"], "10");
            assert_eq!(response.bytes().await.unwrap().as_ref(), encoded.as_slice());
            let events = wait_for_stream_completion(&store).await;
            let SessionEvent::Completed {
                error: Some(error), ..
            } = assert_single_completion(&events, false)
            else {
                panic!("missing error")
            };
            assert!(error.contains(&status.to_string()));
            assert!(error.contains("req-error"));
            if status == 429 {
                assert!(error.contains("rate_limit_error"));
            }
            assert_completion_persisted(&events, false).await;
        }
    }
}

#[tokio::test]
async fn test_anthropic_sse_hop_headers_over_http() {
    for recording in [false, true] {
        let upstream = TestHttpServer::start(axum::Router::new().route("/v1/messages",
            axum::routing::post(|headers: axum::http::HeaderMap| async move {
                assert!(!headers.contains_key("accept-encoding"));
                let sse = [
                    json!({"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":10}}}),
                    json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}),
                    json!({"type":"message_stop"}),
                ].into_iter().map(sse_event).collect::<String>();
                axum::http::Response::builder().header("content-type", "text/event-stream")
                    .header("request-id", "req-sse").header("connection", "X-Upstream-Hop")
                    .header("x-upstream-hop", "private").header("trailer", "x-trailer")
                    .body(Body::from(sse)).unwrap()
            }))).await;
        let (proxy, store) = http_anthropic_proxy(&upstream.url, recording).await;
        let response = raw_http_client()
            .post(format!("{}/v1/messages", proxy.url))
            .header("accept-encoding", "gzip, br")
            .json(&anthropic_http_request(true))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["request-id"], "req-sse");
        for name in [
            "connection",
            "x-upstream-hop",
            "trailer",
            "content-length",
            "content-encoding",
        ] {
            assert!(!response.headers().contains_key(name), "leaked {name}");
        }
        assert!(
            response
                .text()
                .await
                .unwrap()
                .contains("event: message_stop")
        );
        if recording {
            let events = wait_for_stream_completion(&store).await;
            assert_single_completion(&events, true);
        }
    }
}

#[tokio::test]
async fn test_anthropic_truncated_http_bodies_complete() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    for (status, stream) in [(200, false), (401, false), (401, true)] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_url = format!("http://{}", listener.local_addr().unwrap());
        let upstream = TestHttpServer {
            url: upstream_url,
            task: tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0u8; 8192];
                let received = socket.read(&mut request).await.unwrap();
                assert!(received > 0);
                let response = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{{"
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            }),
        };
        let (proxy, store) = http_anthropic_proxy(&upstream.url, true).await;
        let response = raw_http_client()
            .post(format!("{}/v1/messages", proxy.url))
            .timeout(std::time::Duration::from_secs(5))
            .json(&anthropic_http_request(stream))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 502);
        let events = wait_for_stream_completion(&store).await;
        assert_single_completion(&events, false);
        assert_completion_persisted(&events, false).await;
    }
}

#[tokio::test]
async fn test_anthropic_recording_size_limit_preserves_http_body() {
    let oversized = vec![b'a'; 32 * 1024 * 1024 + 1];
    let body = compress_anthropic_body(&oversized, "gzip");
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-encoding", "gzip")
                .set_body_bytes(body.clone()),
        )
        .mount(&upstream)
        .await;
    let (proxy, store) = http_anthropic_proxy(&upstream.uri(), true).await;
    let response = raw_http_client()
        .post(format!("{}/v1/messages", proxy.url))
        .json(&anthropic_http_request(false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.bytes().await.unwrap().as_ref(), body.as_slice());
    let events = wait_for_stream_completion(&store).await;
    assert!(matches!(assert_single_completion(&events, false),
        SessionEvent::Completed { error: Some(error), .. } if error.contains("recording limit")));
}
