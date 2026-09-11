# DementiaCare 🧠

DementiaCare is a full-stack application designed to support dementia patients and their caregivers. It features a fast, responsive patient screen built with React and Vite, powered by a robust Python FastAPI backend and a MongoDB database for event tracking and management.

---

## 🌟 Features

- **Patient Screen Interface**: A dedicated frontend application (`dementiacare-patient-screen`) tailored for ease of use and accessibility.
- **DementiaCare OS API**: A fast and reliable backend REST API built using FastAPI.
- **Event Tracking System**: Built-in support for event management and storage using MongoDB.
- **Modern Tech Stack**: Utilizes React, Vite, FastAPI, and PyMongo for a highly performant and scalable architecture.

## 🛠️ Tech Stack

### Frontend
- **Framework:** [React 18](https://reactjs.org/)
- **Build Tool:** [Vite](https://vitejs.dev/)

### Backend
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Language:** Python 3.x
- **Database Driver:** PyMongo

### Database
- **Database:** MongoDB

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed on your system:
- **Node.js** (v14 or higher)
- **Python** (v3.8 or higher)
- **MongoDB** (Local instance or Atlas cluster)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DementiaCare.git
   cd DementiaCare
   ```

2. **Backend Setup:**
   Open a terminal and navigate to the backend directory to install dependencies and start the API server:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
   *The FastAPI server will typically run on `http://localhost:8000`. You can access the automatic interactive API documentation at `http://localhost:8000/docs`.*

   > **Note:** `main.py` currently lives at the repository root (not in `backend/`), and it imports `config.db` / `models.event`, which resolve only when `backend/` is on the Python path. Until that is tidied up (see `fall_detection/FOLLOWUPS.md`), start the API from the repo root with:
   > ```bash
   > # from the repository root
   > PYTHONPATH=backend uvicorn main:app --reload      # bash
   > $env:PYTHONPATH="backend"; uvicorn main:app --reload   # PowerShell
   > ```
   > A `.env` file at the repo root must define `MONGO_URI` (and `GEMINI_API_KEY` if used).

3. **Frontend Setup:**
   Open a new terminal window, navigate to the frontend directory, install the required NPM packages, and start the development server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   *The Vite development server will typically start on `http://localhost:5173`.*

---

## 📂 Project Structure

```
DementiaCare/
├── backend/                  # FastAPI backend application
│   ├── config/               # Database and app configurations
│   ├── models/               # Data models (event.py: the Event schema)
│   ├── routes/               # API endpoints
│   └── requirements.txt      # Python dependencies
├── main.py                   # FastAPI application entry point (repo root)
├── fall_detection/           # AI fall-detection subsystem (webcam -> /events)
│   ├── config/               # parameters + feature spec
│   ├── common/ features/     # shared pose wrapper + feature extraction
│   ├── data_prep/ splits/ windowing/   # URFD -> feature table -> windows
│   ├── training/ evaluation/ # Random Forest + GRU + subject-wise metrics
│   ├── inference/            # FallDetector, state machine, events client, run_webcam
│   └── tests/                # pytest suite (no webcam/DB/GPU needed)
└── frontend/                 # React frontend application
    ├── src/                  # React components and source code
    ├── index.html            # Main HTML file
    ├── package.json          # Node.js dependencies and scripts
    └── vite.config.js        # Vite configuration
```

## 🚑 Fall Detection

`fall_detection/` adds an optional AI fall-detection process: a webcam feed is run
through MediaPipe Pose, engineered pose/movement features are collected into short
temporal windows, a GRU (Random Forest baseline) estimates fall probability, a
smoother + `NORMAL → SUSPECTED → CONFIRMED → COOLDOWN` state machine confirms real
falls, and a confirmed fall is POSTed to `/events` (with a direct-MongoDB fallback).

It runs on a CPU laptop and is trained on the [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html).
See [`fall_detection/README.md`](fall_detection/README.md) for the full setup and
training/inference runbook.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
