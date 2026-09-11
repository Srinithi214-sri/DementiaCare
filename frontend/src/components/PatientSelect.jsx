import { useEffect, useState } from "react";
import { api } from "../services/api";
import { useAuth } from "../contexts/AuthContext";

export default function PatientSelect({ onSelectPatient }) {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { logout, user } = useAuth();

  useEffect(() => {
    async function fetchPatients() {
      try {
        const data = await api.getPatients();
        setPatients(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchPatients();
  }, []);

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "2rem", fontFamily: "sans-serif" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
        <h2>Authorized Patients</h2>
        <div>
          <span style={{ marginRight: "1rem", color: "#666" }}>Logged in as {user?.name}</span>
          <button onClick={logout} style={{ padding: "0.5rem 1rem", cursor: "pointer" }}>Logout</button>
        </div>
      </header>

      {loading && <p>Loading patients...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      
      {!loading && !error && patients.length === 0 && (
        <p>No patients are currently assigned to you.</p>
      )}

      <div style={{ display: "grid", gap: "1rem" }}>
        {patients.map(p => (
          <div 
            key={p.id} 
            style={{ 
              padding: "1.5rem", 
              border: "1px solid #ddd", 
              borderRadius: "8px", 
              display: "flex", 
              justifyContent: "space-between",
              alignItems: "center",
              backgroundColor: "#f9f9f9"
            }}
          >
            <div>
              <h3 style={{ margin: "0 0 0.5rem 0" }}>{p.name} {p.preferred_name ? `(${p.preferred_name})` : ""}</h3>
              <p style={{ margin: 0, color: "#666", fontSize: "0.9rem" }}>Timezone: {p.timezone} | Language: {p.language}</p>
            </div>
            <button 
              onClick={() => onSelectPatient(p)}
              style={{ padding: "0.5rem 1rem", background: "#28a745", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
            >
              Select
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
