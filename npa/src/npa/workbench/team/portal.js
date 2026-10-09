// Read live permissions and run access through the authenticated same-origin API.
"use strict";
let csrf = "";
const element = (id) => document.getElementById(id);

async function loginMethods() {
  const response = await fetch("/auth/methods", { cache: "no-store" });
  if (!response.ok) throw new Error("Sign-in options are unavailable.");
  const methods = await response.json();
  element("key-login").hidden = !methods.access_key;
  element("organization-login").hidden = !methods.oidc;
}

element("key-login").addEventListener("submit", async (event) => {
  event.preventDefault();
  const credential = element("access-key");
  const body = JSON.stringify({ access_key: credential.value.trim() });
  credential.value = "";
  try {
    const response = await fetch("/auth/key", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Workbench-Login": "1" },
      body,
    });
    if (!response.ok) throw new Error("Sign-in failed. Check your access key with your administrator.");
    await refreshAccess();
  } catch (error) {
    showError(error);
  }
});

async function refreshAccess() {
  element("message").textContent = "";
  const response = await fetch("/v1/me", { cache: "no-store" });
  if (response.status === 401) {
    element("signed-out").hidden = false;
    element("signed-in").hidden = true;
    csrf = "";
    return;
  }
  if (!response.ok)
    throw new Error(
      "Your access is unavailable. Ask your administrator to check your membership.",
    );
  const profile = await response.json();
  csrf = profile.csrf;
  element("signed-out").hidden = true;
  element("signed-in").hidden = false;
  element("person").textContent = profile.display_name || profile.subject;
  element("groups").replaceChildren(...profile.groups.map(groupTag));
  element("workspaces").replaceChildren(
    ...profile.workspaces.map(workspaceCard),
  );
  if (!profile.workspaces.length)
    element("workspaces").textContent =
      "You are signed in, but have no Workbench workspace grants.";
  const { csrf: unused, ...identity } = profile;
  element("identity-details").textContent = JSON.stringify(identity, null, 2);
}

function groupTag(name) {
  const tag = document.createElement("span");
  tag.className = "badge";
  tag.textContent = name;
  return tag;
}

function workspaceCard(workspace) {
  const card = document.createElement("article");
  card.className = "card";
  const title = document.createElement("h3");
  title.textContent = workspace.name;
  const role = document.createElement("p");
  role.textContent = `Role: ${workspace.role}`;
  const allocation = document.createElement("p");
  allocation.textContent = workspace.allocated
    ? `Personal GPU limit: ${workspace.gpu_limit}. Clusters: ${Object.keys(workspace.clusters).join(", ") || "none"}.`
    : "Membership active · awaiting cluster and storage allocation";
  const button = document.createElement("button");
  button.className = "secondary";
  button.textContent = "Show my runs";
  button.addEventListener("click", () => {
    element("workspace").value = workspace.name;
    showRuns(workspace.name).catch(showError);
  });
  card.append(title, role, allocation, button);
  return card;
}

async function showRuns(workspace) {
  const response = await fetch(
    `/v1/runs?workspace=${encodeURIComponent(workspace)}`,
    { cache: "no-store" },
  );
  const payload = await response.json();
  element("access-result").textContent =
    `HTTP ${response.status}\n${JSON.stringify(payload, null, 2)}`;
}

function showError(error) {
  element("message").textContent = error.message;
}

element("refresh").addEventListener("click", () =>
  refreshAccess().catch(showError),
);
element("check-access").addEventListener("submit", (event) => {
  event.preventDefault();
  showRuns(element("workspace").value.trim()).catch(showError);
});
element("sign-out").addEventListener("click", async () => {
  try {
    const response = await fetch("/auth/logout", {
      method: "POST",
      headers: { "X-Workbench-CSRF": csrf },
    });
    if (!response.ok)
      throw new Error("Sign-out failed. Refresh the page and try again.");
    location.assign("/");
  } catch (error) {
    showError(error);
  }
});
loginMethods().then(refreshAccess).catch(showError);
