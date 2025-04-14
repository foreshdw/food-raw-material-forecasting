import React from "react";
import Sidebar from "../components/Sidebar";
import Dashboard from "../components/Dashboard";

const Home = () => {
    return (
        <div className="flex h-screen flex-col md:flex-row bg-gray-100">
            <Sidebar />

            <div className="flex-1 overflow-auto p-2 md:ml-64">
                <Dashboard />
            </div>
        </div>
    );
};

export default Home;
