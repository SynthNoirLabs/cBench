"""Tests for the API handler."""

from handler import Request, handle_request, reset_store


def setup_function():
    reset_store()


def test_create_user():
    req = Request("POST", "/users", body='{"name": "Alice", "email": "alice@example.com"}')
    resp = handle_request(req)
    assert resp.status == 201
    data = resp.json()
    assert data["name"] == "Alice"
    assert data["email"] == "alice@example.com"
    assert "id" in data


def test_list_users():
    handle_request(Request("POST", "/users", body='{"name": "Alice", "email": "a@b.com"}'))
    handle_request(Request("POST", "/users", body='{"name": "Bob", "email": "b@b.com"}'))
    resp = handle_request(Request("GET", "/users"))
    assert resp.status == 200
    assert len(resp.json()["users"]) == 2


def test_get_user():
    handle_request(Request("POST", "/users", body='{"name": "Alice", "email": "a@b.com"}'))
    resp = handle_request(Request("GET", "/users/1"))
    assert resp.status == 200
    assert resp.json()["name"] == "Alice"


def test_get_nonexistent_user():
    resp = handle_request(Request("GET", "/users/999"))
    assert resp.status == 404


def test_update_user():
    handle_request(Request("POST", "/users", body='{"name": "Alice", "email": "a@b.com"}'))
    req = Request("PUT", "/users/1", body='{"name": "Alice Updated", "email": "new@b.com"}')
    resp = handle_request(req)
    assert resp.status == 200
    assert resp.json()["name"] == "Alice Updated"


def test_delete_user():
    handle_request(Request("POST", "/users", body='{"name": "Alice", "email": "a@b.com"}'))
    resp = handle_request(Request("DELETE", "/users/1"))
    assert resp.status == 200
    # Verify user is actually gone
    resp2 = handle_request(Request("GET", "/users/1"))
    assert resp2.status == 404


def test_delete_nonexistent_user():
    """Deleting a user that doesn't exist should return 404, not 200."""
    resp = handle_request(Request("DELETE", "/users/999"))
    assert resp.status == 404


def test_create_user_malformed_json():
    """Malformed JSON should return 400, not crash."""
    req = Request("POST", "/users", body="not valid json{{{")
    resp = handle_request(req)
    assert resp.status == 400
    assert "error" in resp.json()


def test_update_user_partial_body():
    """PUT with missing fields should return 400, not crash."""
    handle_request(Request("POST", "/users", body='{"name": "Alice", "email": "a@b.com"}'))
    req = Request("PUT", "/users/1", body='{"name": "Alice Updated"}')
    resp = handle_request(req)
    assert resp.status == 400
    assert "error" in resp.json()


def test_not_found_route():
    resp = handle_request(Request("GET", "/unknown"))
    assert resp.status == 404
