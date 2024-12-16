// script.js

// Fungsi untuk mengupdate clock
function upClock() {
    let now = new Date();
    let hari = now.getDay(),
        bulan = now.getMonth(),
        tanggal = now.getDate(),
        tahun = now.getFullYear(),
        jam = now.getHours(),
        menit = now.getMinutes(),
        detik = now.getSeconds(),
        periode = "WIB";

    // Tambahkan fungsi pad untuk menambahkan nol di depan angka
    Number.prototype.pad = function(digitals) {
        let n = this.toString();
        while (n.length < digitals) {
            n = "0" + n;
        }
        return n;
    }

    let bulans = ["JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI", "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER"];
    let minggu = ["MINGGU,", "SENIN,", "SELASA,", "RABU,", "KAMIS,", "JUMAT,", "SABTU,"];
    let ids = ["hr", "tg", "bln", "thn", "jam", "mnt", "dtk", "prd"];
    let value = [minggu[hari], tanggal.pad(2), bulans[bulan], tahun, jam.pad(2), menit.pad(2), detik.pad(2), periode];
    for (let i = 0; i < ids.length; i++) {
        document.getElementById(ids[i]).firstChild.nodeValue = value[i];
    }
}

function initClock() {
    upClock();
    window.setInterval(upClock, 1000);
}

// Navbar scroll effect
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    navbar.style.top = `${window.scrollY}px`;
});
