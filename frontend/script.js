const $ = (s) => document.querySelector(s);
const usd = (n) => "$" + Math.round(n).toLocaleString("en-US");
let chart;

$("#preset").addEventListener("change", (e) => {
  if (!e.target.value) return;
  const [lat, lon, ocean] = e.target.value.split(",");
  $("[name=latitude]").value = lat;
  $("[name=longitude]").value = lon;
  $("[name=ocean_proximity]").value = ocean;
});

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const payload = {
    longitude: +f.get("longitude"),
    latitude: +f.get("latitude"),
    housing_median_age: +f.get("housing_median_age"),
    total_rooms: +f.get("total_rooms"),
    total_bedrooms: +f.get("total_bedrooms"),
    population: +f.get("population"),
    households: +f.get("households"),
    median_income: +f.get("income_usd") / 10000, // dataset: satuan $10.000
    ocean_proximity: f.get("ocean_proximity"),
  };
  const btn = $("#btn"), res = $("#result"), err = $("#error");
  btn.disabled = true; err.classList.add("hidden");
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : "Input tidak valid: " +
        d.map((x) => x.loc.slice(-1)[0] + " " + x.msg).join("; "));
    }
    res.innerHTML = `<div>Perkiraan harga median rumah</div><div class="price">${usd(data.predicted_value)}</div>`;
    res.classList.remove("hidden");
    loadHistory();
  } catch (ex) {
    res.classList.add("hidden");
    err.textContent = ex.message; err.classList.remove("hidden");
  } finally { btn.disabled = false; }
});

async function loadHistory() {
  const rows = await (await fetch("/api/history?limit=15")).json();
  $("#history").innerHTML = rows.map((r) => `
    <tr><td>${r.created_at}</td><td>${r.latitude.toFixed(2)}, ${r.longitude.toFixed(2)}</td>
    <td>${r.housing_median_age}</td><td>${r.total_rooms}</td><td>${r.population}</td>
    <td>${usd(r.median_income * 10000)}</td><td>${r.ocean_proximity}</td>
    <td><b>${usd(r.predicted_value)}</b></td></tr>`).join("") ||
    `<tr><td colspan="8">Belum ada prediksi.</td></tr>`;
}

$("#clear").addEventListener("click", async () => {
  if (!confirm("Hapus semua riwayat?")) return;
  await fetch("/api/history", { method: "DELETE" });
  loadHistory();
});

async function loadMetrics() {
  const box = $("#metrics");
  const r = await fetch("/api/metrics");
  if (!r.ok) { box.textContent = "Model belum dilatih (jalankan train.py)."; return; }
  const m = await r.json();
  box.innerHTML = `
    <div><b>${usd(m.test_mae_usd)}</b><span>MAE</span></div>
    <div><b>${usd(m.test_rmse_usd)}</b><span>RMSE</span></div>
    <div><b>${m.test_r2.toFixed(3)}</b><span>R²</span></div>`;
  const h = m.history, labels = h.loss.map((_, i) => i + 1);
  chart?.destroy();
  chart = new Chart($("#chart"), {
    type: "line",
    data: { labels, datasets: [
      { label: "Train loss (MSE)", data: h.loss, borderColor: "#2f6fed", pointRadius: 0 },
      { label: "Val loss (MSE)", data: h.val_loss, borderColor: "#e8710a", pointRadius: 0 },
    ]},
    options: { scales: { y: { type: "logarithmic" }, x: { title: { display: true, text: "Epoch" } } } },
  });
}

loadMetrics(); loadHistory();
