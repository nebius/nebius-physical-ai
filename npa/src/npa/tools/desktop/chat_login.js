/* Exchange the password once, keeping errors and deep links on the login page. */
"use strict";
const form = document.querySelector("form");
const error = document.querySelector("#error");
const destination = new URLSearchParams(location.search).get("next") || "/chat/";
form.elements.next.value = destination + location.hash;
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  error.hidden = true;
  try {
    const response = await fetch(form.action, {
      method: "POST", credentials: "same-origin",
      body: new URLSearchParams(new FormData(form)),
    });
    if (!response.ok) throw new Error(response.status === 401
      ? "The username or password is incorrect."
      : "Sign-in is unavailable. Please try again shortly.");
    // The server validates the return destination before issuing the redirect.
    const target = new URL(response.url);
    if (location.hash) target.hash = location.hash;
    location.replace(target.href);
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
    button.disabled = false;
  }
});
