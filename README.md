# Food Raw Material Forecasting for Final Project
This is my final project (skripsi) for graduation requirements at Universitas Pembangunan Nasional "Veteran" Jakarta. The project is a food raw material forecasting system that utilizes machine learning (LSTM) to predict the required raw materials based on past sales data.

The system consists of:

- Backend: Developed using Flask (Python) for handling API requests and running the forecasting model.

- Frontend: Built with React.js for an interactive and user-friendly UI.

- Styling: Uses Tailwind CSS for modern and responsive design.

- Machine Learning: Implements Long Short-Term Memory (LSTM) to forecast the demand for food raw materials accurately.

## 🛠 Features
✅ Forecasts food raw material needs for the next month based on historical sales data.

✅ Uses LSTM for accurate time-series predictions.

✅ Provides an interactive dashboard for visualization.

✅ REST API for seamless frontend-backend integration.

✅ Responsive and modern UI with Tailwind CSS.

## 📂 Project Structure
```sh
📦 web-app
 ┣ 📂 backend  # Flask-based API and ML model
 ┃ ┣ 📂 models  # Trained LSTM model
 ┃ ┣ 📂 routes  # API endpoints
 ┃ ┣ 📜 app.py  # Main backend application
 ┣ 📂 frontend  # React.js UI
 ┃ ┣ 📜 src/components  # UI components
 ┃ ┣ 📜 src/pages  # Main pages
 ┣ 📜 README.md  # Project documentation
```
---
⚙️ How to Run the Project

Follow these steps to set up and run the project locally:

### 1️⃣ Clone the repository
```sh
git clone https://github.com/foreshdw/skripsi-food-raw-material-forecasting.git
cd skripsi-food-raw-material-forecasting
```

### 2️⃣ Setup Backend
1. Navigate to the backend folder
```sh
cd backend
```
2. Create a virtual environment
```sh
python -m venv venv
```
3. Activate the virtual environment
- Windows
```sh
venv\Scripts\activate
```
- Mac/Linux
```sh
source venv/bin/activate
```
4. Install required dependencies
Since there is no **requirements.txt**, install the necessary packages manually:
```sh
pip install flask flask-cors tensorflow numpy pandas
```
5. Run the backend
```sh
python app.py
```

### 3️⃣ Setup Frontend
```sh
cd frontend
npm install
npm start
```

### 🎯 Future Improvements
🔹 Optimize the LSTM model for better accuracy.

🔹 Compare LSTM with other machine learning algorithms.

🔹 Enhance the dashboard with more data visualization features.

### 📌 Developed as my final thesis project at UPN "Veteran" Jakarta.

