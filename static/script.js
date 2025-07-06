document.addEventListener('DOMContentLoaded', () => {

    // --- TRANSLATION --- //
    const translations = {
        en: {
            title: "Real-Time Trading Dashboard",
            dashboard_title: "Trading Dashboard",
            toggle_theme: "Toggle Theme",
            header_symbol: "Symbol",
            header_latest_close: "Latest Close",
            header_bband_advice: "BBand Advice",
            header_bband_stop_loss: "BBand Stop Loss",
            header_rsi: "RSI",
            header_macd: "MACD",
            header_ema: "EMA",
            header_cci: "CCI",
            header_adx: "ADX",
            header_stoch_k: "Stoch %K",
            header_atr: "ATR",
            header_bbw: "BBW",
            BBAND_BUY: "Buy",
            BBAND_SELL: "Sell",
            NEUTRAL: "Neutral",
            forex_category_title: "Forex",
            futures_category_title: "Futures",
            "AUD Pairs": "AUD Pairs",
            "EUR Pairs": "EUR Pairs",
            "GBP Pairs": "GBP Pairs",
            "USD Pairs": "USD Pairs",
            "CAD Pairs": "CAD Pairs",
            "CHF Pairs": "CHF Pairs",
            "NZD Pairs": "NZD Pairs",
            "Other Forex": "Other Forex Pairs",
            contact_title: "Contact Us",
            contact_email: "Email: @gmail.com",
            contact_instagram: "Instagram: @",
            current_time_label: "Current Time: ",
            search_placeholder: "Search Symbol..."
        }
    };

    let currentLang = localStorage.getItem('lang') || 'en';
    const socket = io();
    let myChart = null; // Singleton chart instance
    let currentChartSymbol = null; // Stores the symbol of the currently displayed chart

    // --- DOM ELEMENTS --- //
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const currentTimeElement = document.getElementById('current-time');
    const forexSubcategoriesContainer = document.getElementById('forex-subcategories-container');
    const futuresTableBody = document.getElementById('futures-table-body');
    const searchInput = document.getElementById('search-input');
    
    let allPredictionsData = {};

    // Get chart modal and close button elements once on DOMContentLoaded
    const chartModal = document.getElementById('chart-modal');
    const closeChartBtn = document.getElementById('close-chart-btn');

    // Critical check: if these core elements are not found, something is fundamentally wrong with the HTML structure or loading.
    if (!chartModal) {
        console.error("CRITICAL ERROR: 'chart-modal' element not found. Chart functionality will be disabled.");
        // Optionally, disable related functionality or show a user-friendly message.
    }
    if (!closeChartBtn) {
        console.error("CRITICAL ERROR: 'close-chart-btn' element not found. Close button functionality will be disabled.");
    }

    // --- HELPER FUNCTIONS --- //
    function formatValue(value, symbol, defaultDecimalPlaces = 5) {
        if (value === null || value === undefined) return 'N/A';
        const num = parseFloat(value);
        if (isNaN(num)) return 'N/A';
        return num.toFixed(defaultDecimalPlaces);
    }

    function showChartModal() {
        if (chartModal) { // Use the globally scoped chartModal
            chartModal.style.display = 'flex';
            if (myChart) {
                myChart.resize(); // Ensure chart is resized correctly when shown
            }
        } else {
            console.error("Error: chartModal element is null in showChartModal. This should not happen if HTML is correct.");
        }
    }

    function hideChartModal() {
        if (chartModal) { // Use the globally scoped chartModal
            chartModal.style.display = 'none';
        } else {
            console.error("Error: chartModal element is null in hideChartModal. This should not happen if HTML is correct.");
        }
    }

    async function fetchChartData(symbol) {
        try {
            const response = await fetch(`/chart_data?symbol=${symbol}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching chart data:', error);
            return null;
        }
    }

    function updateChart(symbol, chartData) {
        const dates = chartData.map(item => item.time);
        const values = chartData.map(item => [item.open, item.close, item.low, item.high]);
        const predictionPrice = chartData.length > 0 ? chartData[chartData.length - 1].prediction_price : null;

        myChart.setOption({
            title: {
                text: `${symbol} Candlestick Chart`
            },
            xAxis: {
                data: dates
            },
            series: [{
                name: symbol,
                data: values,
                ...(predictionPrice && {
                    markLine: {
                        symbol: ['none', 'none'],
                        data: [{
                            name: 'Prediction',
                            yAxis: predictionPrice,
                            lineStyle: { color: '#FF00FF', type: 'solid' },
                            label: { formatter: `Prediction: ${formatValue(predictionPrice, symbol)}` }
                        }]
                    }
                })
            }]
        });
    }

    function initChart(theme) {
        const chartDom = document.getElementById('chart-container');
        if (!chartDom) {
            console.error("Error: chart-container element not found!");
            return;
        }
        myChart = echarts.init(chartDom, theme);

        const option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                formatter: function (params) {
                    const data = params[0];
                    return `${data.name}<br/>
                        Open: ${formatValue(data.value[1])}<br/>
                        Close: ${formatValue(data.value[2])}<br/>
                        Low: ${formatValue(data.value[3])}<br/>
                        High: ${formatValue(data.value[4])}`;
                }
            },
            grid: { left: '10%', right: '10%', bottom: '15%' },
            xAxis: {
                type: 'category',
                scale: true,
                boundaryGap: false,
                axisLine: { onZero: false },
                splitLine: { show: false },
                min: 'dataMin',
                max: 'dataMax'
            },
            yAxis: { scale: true, splitArea: { show: true } },
            dataZoom: [
                { type: 'inside', start: 50, end: 100 },
                { show: true, type: 'slider', top: '90%', start: 50, end: 100 }
            ],
            series: [{
                type: 'candlestick',
                itemStyle: {
                    color: '#ec0000',
                    color0: '#00da3c',
                    borderColor: '#8A0000',
                    borderColor0: '#008F28'
                }
            }]
        };

        myChart.setOption(option);
        window.addEventListener('resize', () => myChart.resize());
    }

    function getAdviceClass(advice) {
        if (advice === 'BBAND_BUY') return 'advice-buy';
        if (advice === 'BBAND_SELL') return 'advice-sell';
        return 'advice-neutral';
    }

    function createTableHeaders() {
        return `<thead><tr>
            <th data-translate="header_symbol">${translations[currentLang]['header_symbol']}</th>
            <th data-translate="header_latest_close">${translations[currentLang]['header_latest_close']}</th>
            <th data-translate="header_bband_advice">${translations[currentLang]['header_bband_advice']}</th>
            <th data-translate="header_bband_stop_loss">${translations[currentLang]['header_bband_stop_loss']}</th>
            <th data-translate="header_rsi">${translations[currentLang]['header_rsi']}</th>
            <th data-translate="header_macd">${translations[currentLang]['header_macd']}</th>
            <th data-translate="header_ema">${translations[currentLang]['header_ema']}</th>
            <th data-translate="header_cci">${translations[currentLang]['header_cci']}</th>
            <th data-translate="header_adx">${translations[currentLang]['header_adx']}</th>
            <th data-translate="header_stoch_k">${translations[currentLang]['header_stoch_k']}</th>
            <th data-translate="header_atr">${translations[currentLang]['header_atr']}</th>
            <th data-translate="header_bbw">${translations[currentLang]['header_bbw']}</th>
        </tr></thead>`;
    }

    function populateTable(tableBody, data) {
        tableBody.innerHTML = '';
        for (const [symbol, values] of Object.entries(data)) {
            const row = document.createElement('tr');
            row.dataset.symbol = symbol;
            const translatedAdvice = translations[currentLang][values.bband_advice] || values.bband_advice;
            row.innerHTML = `
                <td>${symbol}</td>
                <td>${formatValue(values.latest_close, symbol)}</td>
                <td class="${getAdviceClass(values.bband_advice)}">${translatedAdvice}</td>
                <td>${formatValue(values.bband_stop_loss, symbol)}</td>
                <td>${formatValue(values.rsi, symbol)}</td>
                <td>${formatValue(values.macd, symbol)}</td>
                <td>${formatValue(values.ema, symbol)}</td>
                <td>${formatValue(values.cci, symbol)}</td>
                <td>${formatValue(values.adx, symbol)}</td>
                <td>${formatValue(values.stoch_k, symbol)}</td>
                <td>${formatValue(values.atr, symbol)}</td>
                <td>${formatValue(values.bbw, symbol)}</td>
            `;
            tableBody.appendChild(row);
        }
    }

    function filterAndDisplayPredictions(dataToDisplay) {
        const searchTerm = searchInput.value.toLowerCase();
        forexSubcategoriesContainer.innerHTML = '';
        futuresTableBody.innerHTML = '';

        if (dataToDisplay.Futures) {
            const filteredFutures = Object.fromEntries(Object.entries(dataToDisplay.Futures).filter(([s]) => s.toLowerCase().includes(searchTerm)));
            populateTable(futuresTableBody, filteredFutures);
        }

        if (dataToDisplay.Forex) {
            for (const [subCategory, symbolsData] of Object.entries(dataToDisplay.Forex)) {
                const filteredSymbols = Object.fromEntries(Object.entries(symbolsData).filter(([s]) => s.toLowerCase().includes(searchTerm)));
                if (Object.keys(filteredSymbols).length > 0) {
                    const subCatDiv = document.createElement('div');
                    subCatDiv.className = 'subcategory-section';
                    subCatDiv.innerHTML = `<h3 data-translate="${subCategory}">${translations[currentLang][subCategory] || subCategory}</h3><div class="table-responsive-wrapper"><table class="prediction-table">${createTableHeaders()}<tbody id="${subCategory.replace(/\s/g, '-').toLowerCase()}-table-body"></tbody></table></div>`;
                    forexSubcategoriesContainer.appendChild(subCatDiv);
                    populateTable(document.getElementById(`${subCategory.replace(/\s/g, '-').toLowerCase()}-table-body`), filteredSymbols);
                }
            }
        }
    }

    // --- MAIN FUNCTIONS --- //
    function setLanguage(lang) {
        currentLang = lang;
        localStorage.setItem('lang', lang);
        document.querySelectorAll('[data-translate]').forEach(el => {
            const key = el.getAttribute('data-translate');
            if (el.hasAttribute('data-translate-placeholder')) {
                el.placeholder = translations[lang][el.getAttribute('data-translate-placeholder')] || el.getAttribute('data-translate-placeholder');
            } else {
                el.textContent = translations[lang][key] || key;
            }
        });
        document.getElementById('lang-en-btn').classList.toggle('active', lang === 'en');
        document.getElementById('lang-tw-btn').classList.toggle('active', lang === 'zh-TW');
        filterAndDisplayPredictions(allPredictionsData);
    }

    function updateClock() {
        currentTimeElement.textContent = `${translations[currentLang]['current_time_label']}${new Date().toLocaleTimeString()}`;
    }

    async function handleRowClick(event) {
        const row = event.target.closest('tr');
        if (row && row.dataset.symbol) {
            const symbol = row.dataset.symbol;
            const chartData = await fetchChartData(symbol);
            if (chartData) {
                currentChartSymbol = symbol; // Set the current chart symbol
                showChartModal();
                updateChart(symbol, chartData);
            }
        }
    }

    // --- EVENT LISTENERS --- //
    document.getElementById('lang-en-btn').addEventListener('click', () => setLanguage('en'));
    document.getElementById('lang-tw-btn').addEventListener('click', () => setLanguage('zh-TW'));

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            if (myChart) {
                myChart.dispose();
                initChart(newTheme);
            }
        });
    }

    

    document.querySelectorAll('.sidebar-nav a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
        });
    });

    const closeChartBtnElement = document.getElementById('close-chart-btn');
    if (closeChartBtnElement) {
        closeChartBtnElement.addEventListener('click', hideChartModal);
    }

    const chartModalElement = document.getElementById('chart-modal');
    if (chartModalElement) {
        chartModalElement.addEventListener('click', (event) => {
            if (event.target === chartModalElement) { // Check if the click was directly on the modal background
                hideChartModal();
            }
        });
    }

    // Add event listeners to tables for row clicks
    if (forexSubcategoriesContainer) {
        forexSubcategoriesContainer.addEventListener('click', handleRowClick);
    }
    if (futuresTableBody) {
        futuresTableBody.addEventListener('click', handleRowClick);
    }

    // --- INITIALIZATION --- //
    socket.on('data_update', (data) => {
        allPredictionsData = data;
        filterAndDisplayPredictions(allPredictionsData);
    });

    socket.on('chart_data_updated', async (data) => {
        console.log('Chart data updated event received:', data);
        if (currentChartSymbol && chartModal.style.display === 'flex') {
            // If a chart is currently open, re-fetch its data and update it
            const chartData = await fetchChartData(currentChartSymbol);
            if (chartData) {
                updateChart(currentChartSymbol, chartData);
            }
        }
    });

    const initialTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', initialTheme);

    setLanguage(currentLang);
    updateClock();
    setInterval(updateClock, 1000);
    initChart(initialTheme); // Initialize the chart on page load
});