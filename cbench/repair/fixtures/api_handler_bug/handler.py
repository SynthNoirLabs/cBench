"""Simple API request handler with missing error handling."""

from __future__ import annotations

import json
from typing import Any


class Request:
    def __init__(self, method: str, path: str, body: str | None = None, headers: dict[str, str] | None = None):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}


class Response:
    def __init__(self, status: int, body: dict[str, Any] | str, headers: dict[str, str] | None = None):
        self.status = status
        self.body = body
        self.headers = headers or {"Content-Type": "application/json"}

    def json(self) -> dict[str, Any]:
        if isinstance(self.body, dict):
            return self.body
        result: dict[str, Any] = json.loads(self.body)
        return result


# In-memory data store
_users: dict[int, dict[str, Any]] = {}
_next_id = 1


def handle_request(request: Request) -> Response:
    """Route and handle API requests.

    BUG 1: No validation of request body JSON (crashes on malformed JSON)
    BUG 2: DELETE doesn't check if user exists (returns 200 even for missing users)
    BUG 3: PUT doesn't validate that required fields are present
    """
    if request.path == "/users" and request.method == "GET":
        return _list_users()
    elif request.path == "/users" and request.method == "POST":
        return _create_user(request)
    elif request.path.startswith("/users/") and request.method == "GET":
        user_id = int(request.path.split("/")[-1])
        return _get_user(user_id)
    elif request.path.startswith("/users/") and request.method == "PUT":
        user_id = int(request.path.split("/")[-1])
        return _update_user(user_id, request)
    elif request.path.startswith("/users/") and request.method == "DELETE":
        user_id = int(request.path.split("/")[-1])
        return _delete_user(user_id)
    else:
        return Response(404, {"error": "Not found"})


def _list_users() -> Response:
    return Response(200, {"users": list(_users.values())})


def _create_user(request: Request) -> Response:
    global _next_id
    # BUG 1: No try/except for json.loads — crashes on malformed JSON
    data = json.loads(request.body)  # type: ignore[arg-type]
    user = {"id": _next_id, "name": data["name"], "email": data["email"]}
    _users[_next_id] = user
    _next_id += 1
    return Response(201, user)


def _get_user(user_id: int) -> Response:
    if user_id not in _users:
        return Response(404, {"error": f"User {user_id} not found"})
    return Response(200, _users[user_id])


def _update_user(user_id: int, request: Request) -> Response:
    if user_id not in _users:
        return Response(404, {"error": f"User {user_id} not found"})
    # BUG 3: No validation of required fields — KeyError if name/email missing
    data = json.loads(request.body)  # type: ignore[arg-type]
    _users[user_id]["name"] = data["name"]
    _users[user_id]["email"] = data["email"]
    return Response(200, _users[user_id])


def _delete_user(user_id: int) -> Response:
    # BUG 2: Doesn't check if user exists, always returns 200
    _users.pop(user_id, None)
    return Response(200, {"deleted": True})


def reset_store() -> None:
    """Reset the in-memory store (for testing)."""
    global _next_id
    _users.clear()
    _next_id = 1
