const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Helper to retrieve token without exposing it globally
 */
function getToken() {
  return localStorage.getItem("dementiacare_token");
}

/**
 * Clears auth state globally
 */
function clearAuth() {
  localStorage.removeItem("dementiacare_token");
  localStorage.removeItem("dementiacare_user");
  // Dispatch a custom event to notify the AuthContext of 401
  window.dispatchEvent(new Event("auth_unauthorized"));
}

/**
 * Centralized fetch wrapper that auto-injects JWT and handles 401s
 */
async function fetchWithAuth(endpoint, options = {}) {
  const token = getToken();
  
  const headers = new Headers(options.headers || {});
  headers.set("Content-Type", "application/json");
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    clearAuth();
    throw new Error("Unauthorized");
  }

  return response;
}

export const api = {
  // --- AUTH ---
  login: async (username, password) => {
    // Note: OAuth2PasswordRequestForm requires x-www-form-urlencoded
    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    const response = await fetch(`${BASE_URL}/api/v1/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: formData.toString()
    });
    
    if (!response.ok) {
      if (response.status === 401) throw new Error("Incorrect credentials");
      throw new Error("Login failed");
    }
    return response.json();
  },

  getMe: async () => {
    const res = await fetchWithAuth("/api/v1/auth/me");
    if (!res.ok) throw new Error("Failed to fetch user");
    return res.json();
  },

  // --- PATIENTS ---
  getPatients: async () => {
    const res = await fetchWithAuth("/api/v1/patients");
    if (!res.ok) throw new Error("Failed to fetch patients");
    return res.json();
  },

  getPatient: async (id) => {
    const res = await fetchWithAuth(`/api/v1/patients/${id}`);
    if (!res.ok) throw new Error("Failed to fetch patient");
    return res.json();
  },

  // --- MEMORIES ---
  getMemories: async (patientId) => {
    const res = await fetchWithAuth(`/api/v1/patients/${patientId}/memories`);
    if (!res.ok) throw new Error("Failed to fetch memories");
    return res.json();
  },

  // --- EVENTS ---
  getEvents: async (patientId) => {
    const res = await fetchWithAuth(`/api/v1/events?patient_id=${patientId}`);
    if (!res.ok) throw new Error("Failed to fetch events");
    return res.json();
  },

  createEvent: async (patientId, payload = {}, source = "ui", eventType = "user_interaction") => {
    const event = {
      patient_id: patientId,
      source: source,
      event_type: eventType,
      timestamp: new Date().toISOString(),
      payload
    };
    const res = await fetchWithAuth("/api/v1/events", {
      method: "POST",
      body: JSON.stringify(event)
    });
    if (!res.ok) throw new Error("Failed to create event");
    return res.json();
  },

  // --- AGENT ---
  processAgent: async (patientId, eventId) => {
    const res = await fetchWithAuth("/api/v1/agent/process", {
      method: "POST",
      body: JSON.stringify({
        patient_id: patientId,
        event_id: eventId
      })
    });
    if (!res.ok) throw new Error("Agent processing failed");
    return res.json();
  }
};
