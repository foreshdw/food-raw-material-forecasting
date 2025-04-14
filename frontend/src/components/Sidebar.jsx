import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FiHome, FiLogOut } from "react-icons/fi";
import { BsFillHouseFill } from "react-icons/bs"

const Sidebar = () => {
    const navigate = useNavigate();
    const [isOpen, setIsOpen] = useState(window.innerWidth >= 768); // Default terbuka di desktop

    const handleLogout = () => {
        navigate("/");
    };

    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth >= 768) {
                setIsOpen(true); // Pastikan sidebar selalu terbuka di mode desktop
            }
        };

        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return (
        <>
            {/* Tombol Toggle Sidebar di Layar Kecil */}
            <button 
                className="fixed top-5 left-4 z-50 py-2 px-3 bg-black text-white rounded-md md:hidden"
                onClick={() => setIsOpen(!isOpen)}
            >
                ☰
            </button>

            {/* Sidebar */}
            <div 
                className={`fixed top-0 left-0 h-screen bg-white shadow-slate-400 shadow-sm border border-slate-300 transition-transform transform 
                ${isOpen ? "translate-x-0" : "-translate-x-full"} 
                w-4/5 max-w-xs md:w-64 p-4 z-40`} // Lebar sidebar lebih proporsional
            >
                {/* Logo di tengah */}
                <div className="flex justify-center items-center">
                    <img src="logoFotkop.png" alt="Logo Fotkop" className="h-10 md:h-auto w-auto" />
                </div>

                {/* Menu */}
                <ul className="mt-4 space-y-2">
                    <li className="flex items-center gap-2 px-4 py-2 bg-yellow-500 text-black rounded-full cursor-pointer font-medium">
                        <BsFillHouseFill className="text-lg" /> Dasbor
                    </li>
                    <li 
                        className="flex items-center gap-2 px-4 py-2 text-red-500 cursor-pointer hover:bg-gray-300 rounded-full font-medium"
                        onClick={handleLogout}
                    >
                        <FiLogOut className="text-lg" /> Keluar
                    </li>
                </ul>
            </div>
        </>
    );
};

export default Sidebar;
