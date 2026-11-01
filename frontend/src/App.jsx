import { useState } from "react";
import { useAuth } from "./contexts/AuthContext";
import Login from "./components/Login";
import PatientSelect from "./components/PatientSelect";
import PatientProfile from "./components/PatientProfile";
import CompanionScreen from "./components/CompanionScreen";
import "./App.css";

function AppContent() {
  const { isAuthenticated, loading } = useAuth();
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showCompanion, setShowCompanion] = useState(false);

  if (loading) {
    return <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Login />;
  }

  if (!selectedPatient) {
    return <PatientSelect onSelectPatient={setSelectedPatient} />;
  }

  if (showCompanion) {
    return <CompanionScreen patient={selectedPatient} onExit={() => setShowCompanion(false)} />;
  }

  return (
    <PatientProfile 
      patient={selectedPatient} 
      onLaunch={() => setShowCompanion(true)} 
      onBack={() => setSelectedPatient(null)} 
    />
  );
}

export default function App() {
  return (
    <AppContent />
  );
}
