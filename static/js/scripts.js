async function predict() {
    const formData = {
        labor_hours: parseFloat(document.getElementById('labor_hours').value),
        machine_downtime: parseFloat(document.getElementById('machine_downtime').value),
        material_quality: parseFloat(document.getElementById('material_quality').value),
        maintenance_frequency: parseFloat(document.getElementById('maintenance_frequency').value),
        energy_costs: parseFloat(document.getElementById('energy_costs').value)
    };

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });

    const result = await response.json();
    document.getElementById('results').innerHTML = `
        <p><strong>Productivity:</strong> ${result.productivity.toFixed(2)}</p>
        <p><strong>Cost:</strong> ${result.cost.toFixed(2)}</p>
        <p><strong>Profit:</strong> ${result.profit.toFixed(2)}</p>
    `;
}

async function loadPlots() {
    const response = await fetch('/plots');
    const plots = await response.json();

    const plotsContainer = document.getElementById('plots');
    plotsContainer.innerHTML = '';
    plots.forEach(plot => {
        const img = document.createElement('img');
        img.src = plot;
        plotsContainer.appendChild(img);
    });
}

document.addEventListener('DOMContentLoaded', loadPlots);


