import { createContext, useContext, useState, useEffect } from "react";
import { api } from "../services/api";
import { textToSpeech } from "../services/textToSpeech";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("dementiacare_token"));
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("dementiacare_user");
    return saved ? JSON.parse(saved) : null;
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handleUnauthorized = () => {
      setToken(null);
      setUser(null);
    };
    window.addEventListener("auth_unauthorized", handleUnauthorized);
    return () => window.removeEventListener("auth_unauthorized", handleUnauthorized);
  }, []);

  const login = async (email, password) => {
    setLoading(true);
    try {
      const data = await api.login(email, password);
      // NOTE: Storing JWT in localStorage exposes it to XSS. 
      // For Phase 9, this is a known development tradeoff.
      localStorage.setItem("dementiacare_token", data.access_token);
      setToken(data.access_token);
      
      const userData = await api.getMe();
      localStorage.setItem("dementiacare_user", JSON.stringify(userData));
      setUser(userData);
      return true;
    } catch (err) {
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    textToSpeech.stop();
    localStorage.removeItem("dementiacare_token");
    localStorage.removeItem("dementiacare_user");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ token, user, isAuthenticated: !!token, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
