import React, { useState } from "react";
import axios from "axios";

const Dashboard = () => {
    const [selectedMenu, setSelectedMenu] = useState("Americano");
    const [prediction, setPrediction] = useState(null);

    const handlePredict = async () => {
        try {
            const response = await axios.post("http://localhost:5000/predict", { menu: selectedMenu });
            console.log("API Response Data:", response.data);
            setPrediction(response.data);
        } catch (error) {
            console.error("Error fetching prediction:", error);
        }
    };

    return (
        <div className="flex flex-col p-5 w-full">
            <h1 className="text-xl font-medium">Selamat Datang!</h1>
            <h2 className="text-xl font-semibold">Dasbor Prediksi Warung Fotkop</h2>
            
            {/* Statistik */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                <div className="bg-black text-yellow-500 p-4 rounded-2xl text-center md:text-left shadow-sm">
                    <p>Total Menu</p>
                    <h3 className="text-xl font-bold">2</h3>
                </div>
                <div className="bg-black text-yellow-500 p-4 rounded-2xl text-center md:text-left shadow-sm">
                    <p>Total Menu Terjual Bulan Depan</p>
                    <h3 className="text-xl font-bold">{prediction ? `${prediction.jumlah_terjual} Porsi` : "xxx Porsi"}</h3>
                </div>
                <div className="bg-black text-yellow-500 p-4 rounded-2xl text-center md:text-left shadow-sm">
                    <p>Total Bahan Baku Bulan Depan</p>
                    <h3 className="text-xl font-bold">{prediction ? `${Object.values(prediction.bahan_baku).reduce((a, b) => a + b, 0)} gram/ml` : "xxxxx gram/ml"}</h3>
                </div>
            </div>

            {/* Form Prediksi */}
            <div className="bg-white p-6 rounded-2xl mt-4 border border-slate-300 shadow-slate-400 shadow-sm">
                <h1 className="font-bold text-xl mb-2">Prediksi Kebutuhan Bahan Baku Bulan Depan</h1>
                <p className="text-sm text-gray-600 mb-3">
                    Berencana menyusun stok bahan baku lebih akurat? Gunakan fitur prediksi ini untuk memperkirakan kebutuhan bahan baku berdasarkan data penjualan sebelumnya. Cukup pilih menu dan periode waktu, lalu lihat hasilnya.
                </p>

                <label className="block text-base font-semibold text-black ml-1">Pilih Menu</label>
                <div className="relative w-full mt-2">
                    <select
                        className="w-full p-2 pr-12 pl-4 border border-slate-300 rounded-full appearance-none bg-white"
                        value={selectedMenu}
                        onChange={(e) => setSelectedMenu(e.target.value)}
                    >
                        <option>Americano</option>
                        <option>Garlic Fries</option>
                    </select>
                    <div className="absolute inset-y-0 right-3 flex items-center pointer-events-none">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="w-5 h-5 text-gray-700"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                        >
                            <path
                                fillRule="evenodd"
                                d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                                clipRule="evenodd"
                            />
                        </svg>
                    </div>
                </div>
                

                <button
                    onClick={handlePredict}
                    className="w-full bg-yellow-500 text-black font-bold p-2 mt-4 rounded-full"
                >
                    Mulai Prediksi
                </button>
            </div>

            {/* Tabel Hasil Prediksi */}
            {prediction && (
                <div className="bg-white mt-6 p-6 rounded-2xl border border-slate-300 shadow-slate-400 shadow-sm">
                    <h3 className="font-bold text-xl mb-2">Hasil Prediksi Penjualan Bulan Depan</h3>
                    <p>Menu: {prediction.menu}</p>
                    <p>Prediksi Terjual: {prediction.predicted_sold} porsi</p>
                    <h3 className="font-bold text-xl mt-4">Kebutuhan Bahan Baku</h3>
                    <ul>
                        {Object.entries(prediction.ingredients).map(([key, value]) => (
                            <li key={key}>{key}: {value}</li>
                        ))}
                    </ul>
                </div>
            )}

                        {/* Tabel Hasil Prediksi */}
                        <div className="bg-white mt-6 p-6 rounded-2xl border border-slate-300 overflow-x-auto">
                <h3 className="font-bold text-xl mb-2">Hasil Prediksi Penjualan Bulan xxxxx</h3>
                <table className="w-full border-collapse border border-gray-300 text-sm">
                    <thead>
                        <tr className="bg-black text-white">
                            <th className="p-2 text-yellow-500">Menu</th>
                            <th className="p-2 text-yellow-500">Prediksi Terjual</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td className="p-2 text-center">Americano</td>
                            <td className="p-2 text-center">xx Porsi</td>
                        </tr>
                        <tr className="bg-gray-200">
                            <td className="p-2 text-center">Garlic Fries</td>
                            <td className="p-2 text-center">xxx Porsi</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="bg-white mt-6 p-6 rounded-2xl border border-slate-300 overflow-x-auto">
                <h3 className="font-bold text-xl mb-2">Kebutuhan Bahan Baku Bulan xxxxx</h3>
                <table className="w-full border-collapse border border-gray-300 text-sm">
                    <thead>
                        <tr className="bg-black text-white">
                            <th className="p-2 text-yellow-500">Bahan Baku</th>
                            <th className="p-2 text-yellow-500">Prediksi Terjual</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td className="p-2 text-center">Air</td>
                            <td className="p-2 text-center">xxx ml</td>
                        </tr>
                        <tr className="bg-gray-200">
                            <td className="p-2 text-center">Bawang Putih</td>
                            <td className="p-2 text-center">xxx gram</td>
                        </tr>
                        <tr>
                            <td className="p-2 text-center">Biji Kopi Arabika</td>
                            <td className="p-2 text-center">xxx gram</td>
                        </tr>
                        <tr className="bg-gray-200">
                            <td className="p-2 text-center">Garam</td>
                            <td className="p-2 text-center">xxx gram</td>
                        </tr>
                        <tr>
                            <td className="p-2 text-center">Kentang Russet</td>
                            <td className="p-2 text-center">xxx gram</td>
                        </tr>
                        <tr className="bg-gray-200">
                            <td className="p-2 text-center">Lada Hitam</td>
                            <td className="p-2 text-center">xxx gram</td>
                        </tr>
                        <tr>
                            <td className="p-2 text-center">Minyak Goreng</td>
                            <td className="p-2 text-center">xxx ml</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Dashboard;
