import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const LoginPage = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleLogin = (e) => {
        e.preventDefault();

        if (email === "admin@example.com" && password === "password") {
            navigate("/dashboard");
        } else {
            alert("Login gagal! Periksa email dan kata sandi.");
        }
    };

    return (
        <div
            className="flex items-center justify-center h-screen bg-cover bg-center"
            style={{
                backgroundImage:
                    "linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('/background.png')",
            }}
        >
            <div className="bg-white p-10 rounded-2xl shadow-lg min-w-96 max-w-xl">
                <h2 className="text-3xl font-bold text-center mb-2">
                    Selamat Datang
                </h2>
                <p className="text-base text-center text-black mb-6 font-normal">
                    Silahkan masuk untuk melanjutkan
                </p>

                <form onSubmit={handleLogin}>
                    <div className="mb-4">
                        <label className="block text-black">Email</label>
                        <input
                            type="email"
                            placeholder="Tulis email Anda"
                            className="w-full px-4 py-2 border border-slate-300 rounded-full mt-1"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>

                    <div className="mb-4">
                        <label className="block text-black">Kata Sandi</label>
                        <input
                            type="password"
                            placeholder="Kata sandi akun Anda"
                            className="w-full px-4 py-2 border border-slate-300 rounded-full mt-1"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    <div className="flex justify-between items-center mb-4">
                        <label className="flex items-center text-black font-light">
                            <input type="checkbox" className="mr-2" /> Ingat Saya
                        </label>
                        <a href="#" className="text-yellow-500">
                            Lupa Password?
                        </a>
                    </div>

                    <button
                        type="submit"
                        className="w-full bg-yellow-500 text-black font-semibold p-2 rounded-full hover:bg-yellow-600"
                    >
                        Log In
                    </button>
                </form>
            </div>
        </div>
    );
};

export default LoginPage;
