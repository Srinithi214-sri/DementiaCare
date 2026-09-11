import { useEffect, useState } from "react";
import { api } from "../services/api";

export default function PatientProfile({ patient, onLaunch, onBack }) {
  const [profile, setProfile] = useState(patient);
  const [memories, setMemories] = useState([]);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [profileData, memoriesData, eventsData] = await Promise.all([
          api.getPatient(patient.id),
          api.getMemories(patient.id),
          api.getEvents(patient.id)
        ]);
        setProfile(profileData);
        setMemories(memoriesData.memories || []);
        setEvents(eventsData.events || []);
      } catch (err) {
        console.error("Failed to load patient data:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [patient.id]);

  if (loading) return <div style={{ padding: "2rem" }}>Loading patient context...</div>;

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", padding: "2rem", fontFamily: "sans-serif" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem", paddingBottom: "1rem", borderBottom: "1px solid #eee" }}>
        <div>
          <button onClick={onBack} style={{ marginBottom: "1rem", cursor: "pointer" }}>← Back to Patient List</button>
          <h2 style={{ margin: 0 }}>{profile.name}'s Profile</h2>
        </div>
        <button 
          onClick={onLaunch}
          style={{ padding: "1rem 2rem", background: "#007bff", color: "white", border: "none", borderRadius: "8px", fontSize: "1.1rem", fontWeight: "bold", cursor: "pointer" }}
        >
          Launch Companion Screen
        </button>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
        <section>
          <h3>Active Approved Memories</h3>
          {memories.length === 0 ? <p>No approved memories found.</p> : (
            <ul style={{ paddingLeft: "1.5rem" }}>
              {memories.map(m => (
                <li key={m.id} style={{ marginBottom: "0.5rem" }}>
                  <strong>{m.title}</strong>: {m.content}
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <h3>Recent Event History (Limited)</h3>
          {events.length === 0 ? <p>No events recorded.</p> : (
            <div style={{ maxHeight: "400px", overflowY: "auto", border: "1px solid #eee", padding: "1rem", borderRadius: "8px" }}>
              {events.slice(0, 10).map(e => (
                <div key={e.id} style={{ marginBottom: "1rem", paddingBottom: "1rem", borderBottom: "1px solid #f5f5f5" }}>
                  <div style={{ fontSize: "0.8rem", color: "#888" }}>{new Date(e.timestamp).toLocaleString()}</div>
                  <div><strong>{e.event_type}</strong> from {e.source}</div>
                  <pre style={{ margin: "0.5rem 0 0 0", fontSize: "0.8rem", background: "#f9f9f9", padding: "0.5rem" }}>
                    {JSON.stringify(e.payload, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
