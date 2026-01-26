<script>
  import { onMount } from 'svelte';
  import L from 'leaflet';
  import 'leaflet-draw';
  import {
    connectGEE,
    semanticSearch,
    zeroShotSearch,
    copernicusSearch,
    exportResults
  } from './lib/api';

  let geeProjectId = '';
  let geeStatus = 'disconnected';
  let geeMessage = '';

  let activeTab = 'semantic';
  let activeDrawMode = 'aoi';

  let aoiGeojson = null;
  let copernicusQueryGeojson = null;
  let copernicusSearchGeojson = null;

  let query = '';
  let searchType = 'text';
  let referenceImageBase64 = null;
  let startDate = new Date(Date.now() - 1000 * 60 * 60 * 24 * 90).toISOString().slice(0, 10);
  let endDate = new Date().toISOString().slice(0, 10);
  let topK = 10;
  let similarityThreshold = 0.3;
  let resolution = 10;

  let zeroShotStartDate = new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString().slice(0, 10);
  let zeroShotEndDate = new Date().toISOString().slice(0, 10);
  let zeroShotThreshold = 0.65;
  let zeroShotResolution = 10;
  let zeroShotToken = '';
  let zeroShotImageBase64 = null;
  let zeroShotBbox = null;
  let zeroShotCanvas;
  let zeroShotImage = null;
  let zeroShotDrawing = false;
  let zeroShotStart = { x: 0, y: 0 };

  let copernicusStartDate = new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString().slice(0, 10);
  let copernicusEndDate = new Date().toISOString().slice(0, 10);
  let copernicusSensor = 'Sentinel-2';
  let copernicusResolution = 10;
  let copernicusThreshold = 0.5;

  let statusMessage = '';
  let results = [];
  let currentQuery = '';
  let isLoading = false;

  let showHeatmaps = true;
  let heatmapMode = 'Similarity';
  let selectedResults = new Set();

  let map;
  let aoiLayer;
  let queryLayer;
  let searchLayer;

  const drawColors = {
    aoi: '#38bdf8',
    copernicusQuery: '#22c55e',
    copernicusSearch: '#f97316'
  };

  onMount(() => {
    map = L.map('map').setView([20, 0], 2);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    aoiLayer = L.featureGroup().addTo(map);
    queryLayer = L.featureGroup().addTo(map);
    searchLayer = L.featureGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      draw: {
        polygon: {
          allowIntersection: false,
          showArea: true
        },
        rectangle: true,
        circle: false,
        polyline: false,
        marker: false,
        circlemarker: false
      },
      edit: false
    });

    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, (event) => {
      const layer = event.layer;
      const geometry = layer.toGeoJSON().geometry;
      const color = drawColors[activeDrawMode] || '#38bdf8';
      layer.setStyle({ color, fillColor: color, fillOpacity: 0.2 });

      if (activeDrawMode === 'aoi') {
        aoiLayer.clearLayers();
        aoiLayer.addLayer(layer);
        aoiGeojson = geometry;
      } else if (activeDrawMode === 'copernicusQuery') {
        queryLayer.clearLayers();
        queryLayer.addLayer(layer);
        copernicusQueryGeojson = geometry;
      } else if (activeDrawMode === 'copernicusSearch') {
        searchLayer.clearLayers();
        searchLayer.addLayer(layer);
        copernicusSearchGeojson = geometry;
      }
    });
  });

  function setDrawMode(mode) {
    activeDrawMode = mode;
  }

  async function handleConnectGEE() {
    statusMessage = '';
    geeMessage = '';
    try {
      const response = await connectGEE(geeProjectId);
      geeStatus = 'connected';
      geeMessage = response.project_id ? `Project: ${response.project_id}` : 'Connected';
    } catch (error) {
      geeStatus = 'error';
      geeMessage = error.message;
    }
  }

  function handleReferenceImage(event) {
    const file = event.target.files[0];
    if (!file) {
      referenceImageBase64 = null;
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      referenceImageBase64 = reader.result.split(',')[1];
    };
    reader.readAsDataURL(file);
  }

  async function handleSemanticSearch() {
    statusMessage = '';
    if (!aoiGeojson) {
      statusMessage = 'Please draw an area of interest on the map.';
      return;
    }
    if (searchType === 'text' && !query.trim()) {
      statusMessage = 'Please enter a search query.';
      return;
    }
    if (searchType === 'image' && !referenceImageBase64) {
      statusMessage = 'Please upload a reference image.';
      return;
    }

    isLoading = true;
    currentQuery = searchType === 'text' ? query : 'Reference Image';
    try {
      const response = await semanticSearch({
        aoi_geojson: aoiGeojson,
        query,
        start_date: startDate,
        end_date: endDate,
        top_k: topK,
        similarity_threshold: similarityThreshold,
        resolution,
        search_type: searchType,
        reference_image_base64: referenceImageBase64
      });
      results = response.results || [];
      selectedResults = new Set(results.map((_, idx) => idx));
    } catch (error) {
      statusMessage = error.message;
    } finally {
      isLoading = false;
    }
  }

  function handleZeroShotImage(event) {
    const file = event.target.files[0];
    if (!file) {
      zeroShotImageBase64 = null;
      zeroShotImage = null;
      zeroShotBbox = null;
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      zeroShotImageBase64 = dataUrl.split(',')[1];
      const image = new Image();
      image.onload = () => {
        zeroShotImage = image;
        zeroShotBbox = null;
        setTimeout(() => drawZeroShotCanvas(), 0);
      };
      image.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }

  function drawZeroShotCanvas(rect = null) {
    if (!zeroShotCanvas || !zeroShotImage) return;
    const ctx = zeroShotCanvas.getContext('2d');
    zeroShotCanvas.width = zeroShotImage.width;
    zeroShotCanvas.height = zeroShotImage.height;
    ctx.clearRect(0, 0, zeroShotCanvas.width, zeroShotCanvas.height);
    ctx.drawImage(zeroShotImage, 0, 0);
    if (rect) {
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 3;
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
      ctx.fillStyle = 'rgba(249, 115, 22, 0.2)';
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    }
  }

  function handleZeroShotMouseDown(event) {
    if (!zeroShotImage) return;
    const rect = zeroShotCanvas.getBoundingClientRect();
    const scaleX = zeroShotCanvas.width / rect.width;
    const scaleY = zeroShotCanvas.height / rect.height;
    zeroShotDrawing = true;
    zeroShotStart = {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY
    };
  }

  function handleZeroShotMouseMove(event) {
    if (!zeroShotDrawing || !zeroShotImage) return;
    const rect = zeroShotCanvas.getBoundingClientRect();
    const scaleX = zeroShotCanvas.width / rect.width;
    const scaleY = zeroShotCanvas.height / rect.height;
    const current = {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY
    };
    const bbox = {
      x: zeroShotStart.x,
      y: zeroShotStart.y,
      width: current.x - zeroShotStart.x,
      height: current.y - zeroShotStart.y
    };
    drawZeroShotCanvas(bbox);
  }

  function handleZeroShotMouseUp(event) {
    if (!zeroShotDrawing || !zeroShotImage) return;
    zeroShotDrawing = false;
    const rect = zeroShotCanvas.getBoundingClientRect();
    const scaleX = zeroShotCanvas.width / rect.width;
    const scaleY = zeroShotCanvas.height / rect.height;
    const end = {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY
    };
    const bbox = {
      x: Math.min(zeroShotStart.x, end.x),
      y: Math.min(zeroShotStart.y, end.y),
      width: Math.abs(end.x - zeroShotStart.x),
      height: Math.abs(end.y - zeroShotStart.y)
    };
    zeroShotBbox = bbox;
    drawZeroShotCanvas(bbox);
  }

  function resetZeroShotBox() {
    zeroShotBbox = null;
    drawZeroShotCanvas();
  }

  async function handleZeroShotSearch() {
    statusMessage = '';
    if (!aoiGeojson) {
      statusMessage = 'Please draw an area of interest on the map.';
      return;
    }
    if (!zeroShotImageBase64 || !zeroShotBbox) {
      statusMessage = 'Please upload a reference image and draw a bounding box.';
      return;
    }

    isLoading = true;
    currentQuery = 'Zero-Shot Pattern';
    try {
      const response = await zeroShotSearch({
        aoi_geojson: aoiGeojson,
        start_date: zeroShotStartDate,
        end_date: zeroShotEndDate,
        threshold: zeroShotThreshold,
        resolution: zeroShotResolution,
        hf_token: zeroShotToken,
        reference_image_base64: zeroShotImageBase64,
        bbox: [
          Math.round(zeroShotBbox.x),
          Math.round(zeroShotBbox.y),
          Math.round(zeroShotBbox.width),
          Math.round(zeroShotBbox.height)
        ]
      });
      results = response.results || [];
      selectedResults = new Set(results.map((_, idx) => idx));
    } catch (error) {
      statusMessage = error.message;
    } finally {
      isLoading = false;
    }
  }

  async function handleCopernicusSearch() {
    statusMessage = '';
    if (!copernicusQueryGeojson || !copernicusSearchGeojson) {
      statusMessage = 'Please draw both query and search areas on the map.';
      return;
    }

    isLoading = true;
    currentQuery = `CopernicusFM (${copernicusSensor})`;
    try {
      const response = await copernicusSearch({
        query_geom: copernicusQueryGeojson,
        search_geom: copernicusSearchGeojson,
        start_date: copernicusStartDate,
        end_date: copernicusEndDate,
        sensor: copernicusSensor,
        resolution: copernicusResolution,
        threshold: copernicusThreshold
      });
      results = response.results || [];
      selectedResults = new Set(results.map((_, idx) => idx));
    } catch (error) {
      statusMessage = error.message;
    } finally {
      isLoading = false;
    }
  }

  function toggleResultSelection(index) {
    const next = new Set(selectedResults);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    selectedResults = next;
  }

  function selectAllResults() {
    selectedResults = new Set(results.map((_, idx) => idx));
  }

  function deselectAllResults() {
    selectedResults = new Set();
  }

  async function handleExport(format) {
    const selected = results.filter((_, idx) => selectedResults.has(idx));
    if (!selected.length) {
      statusMessage = 'Please select at least one result to export.';
      return;
    }

    try {
      const blob = await exportResults(format, {
        results: selected,
        query: currentQuery || 'EmbeddedEarth Query'
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `embeddedearth.${format === 'geojson' ? 'geojson' : format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      statusMessage = error.message;
    }
  }

  $: hasDinoAttention = results.some((res) => res.dino_attention);
  $: hasPcaMap = results.some((res) => res.pca_map);
</script>

<div class="app">
  <aside class="sidebar">
    <h1>🌍 EmbeddedEarth</h1>

    <section>
      <h3>About</h3>
      <p>
        EmbeddedEarth helps you find features in satellite imagery using natural language.
      </p>
      <ul>
        <li>DOFA-CLIP</li>
        <li>Google Earth Engine</li>
        <li>Sentinel-2 Imagery</li>
      </ul>
    </section>

    <section>
      <h3>GEE Connection</h3>
      <label for="geeProject">GEE Project ID</label>
      <input id="geeProject" type="text" bind:value={geeProjectId} placeholder="my-gee-project" />
      <button class="button-primary" on:click={handleConnectGEE}>Connect</button>
      <p class="status">
        {#if geeStatus === 'connected'}
          ✅ Connected {geeMessage}
        {:else if geeStatus === 'error'}
          ⚠️ {geeMessage}
        {:else}
          ⚠️ Not connected
        {/if}
      </p>
    </section>

    <section>
      <h3>Keyboard Navigation</h3>
      <ul>
        <li>Tab to move between fields</li>
        <li>Enter to submit</li>
        <li>Arrow keys to pan map</li>
        <li>+ / - to zoom map</li>
      </ul>
    </section>
  </aside>

  <main class="main">
    <div class="header">
      <h2>🌍 EmbeddedEarth</h2>
      <p>Semantic Search for Satellite Imagery</p>
    </div>

    <div class="notice">
      <strong>How to Speak “Satellite”</strong>
      <p>
        Use remote sensing vocabulary and spatial descriptions. Example: “high-density residential with
        grid-like urban fabric.” The model performs best with technical land-cover terminology.
      </p>
    </div>

    <div class="content">
      <div class="card">
        <div class="map-wrapper">
          <div id="map"></div>
        </div>
        <div class="form-actions" style="margin-top: 12px;">
          <span>Draw mode:</span>
          <button
            class="button-secondary"
            on:click={() => setDrawMode('aoi')}
          >
            AOI
          </button>
          <button
            class="button-secondary"
            on:click={() => setDrawMode('copernicusQuery')}
          >
            Copernicus Query
          </button>
          <button
            class="button-secondary"
            on:click={() => setDrawMode('copernicusSearch')}
          >
            Copernicus Search
          </button>
        </div>
        <p class="status">
          Active draw mode: {activeDrawMode}
        </p>
      </div>

      <div class="card">
        <div class="tabs">
          <button class={`tab ${activeTab === 'semantic' ? 'active' : ''}`} on:click={() => (activeTab = 'semantic')}>
            💬 Semantic Search
          </button>
          <button class={`tab ${activeTab === 'zeroShot' ? 'active' : ''}`} on:click={() => (activeTab = 'zeroShot')}>
            🎯 Zero-Shot Detection
          </button>
          <button class={`tab ${activeTab === 'copernicus' ? 'active' : ''}`} on:click={() => (activeTab = 'copernicus')}>
            🛰️ Copernicus FM
          </button>
        </div>

        {#if activeTab === 'semantic'}
          <div class="form-grid">
            <label for="searchType">Search Method</label>
            <select id="searchType" bind:value={searchType}>
              <option value="text">Text Query</option>
              <option value="image">Reference Image</option>
            </select>

            {#if searchType === 'text'}
              <label for="searchQuery">Search Query</label>
              <input
                id="searchQuery"
                type="text"
                bind:value={query}
                placeholder="solar panels, deforestation"
              />
            {:else}
              <label for="searchReferenceImage">Reference Image</label>
              <input
                id="searchReferenceImage"
                type="file"
                accept="image/*"
                on:change={handleReferenceImage}
              />
            {/if}

            <div class="form-actions">
              <div style="flex: 1; display: grid; gap: 6px;">
                <label for="semanticStartDate">Start Date</label>
                <input id="semanticStartDate" type="date" bind:value={startDate} />
              </div>
              <div style="flex: 1; display: grid; gap: 6px;">
                <label for="semanticEndDate">End Date</label>
                <input id="semanticEndDate" type="date" bind:value={endDate} />
              </div>
            </div>

            <label for="topK">Top K Results</label>
            <input id="topK" type="number" min="1" max="50" bind:value={topK} />

            <label for="similarityThreshold">Similarity Threshold</label>
            <input
              id="similarityThreshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={similarityThreshold}
            />

            <label for="semanticResolution">Resolution (m/px)</label>
            <input id="semanticResolution" type="number" min="10" max="60" step="10" bind:value={resolution} />

            <button class="button-primary" on:click={handleSemanticSearch} disabled={isLoading}>
              {isLoading ? 'Searching...' : '🚀 Search'}
            </button>
          </div>
        {:else if activeTab === 'zeroShot'}
          <div class="form-grid">
            <label for="zeroShotToken">Hugging Face Token (Optional)</label>
            <input id="zeroShotToken" type="password" bind:value={zeroShotToken} placeholder="HF token" />

            <label for="zeroShotImage">Reference Image</label>
            <input id="zeroShotImage" type="file" accept="image/*" on:change={handleZeroShotImage} />

            {#if zeroShotImage}
              <div class="canvas-wrapper">
                <canvas
                  bind:this={zeroShotCanvas}
                  on:mousedown={handleZeroShotMouseDown}
                  on:mousemove={handleZeroShotMouseMove}
                  on:mouseup={handleZeroShotMouseUp}
                  on:mouseleave={handleZeroShotMouseUp}
                ></canvas>
              </div>
              <div class="form-actions">
                <button class="button-secondary" on:click={resetZeroShotBox}>Clear Box</button>
                {#if zeroShotBbox}
                  <span class="status">Box ready</span>
                {/if}
              </div>
            {/if}

            <div class="form-actions">
              <div style="flex: 1; display: grid; gap: 6px;">
                <label for="zeroShotStartDate">Start Date</label>
                <input id="zeroShotStartDate" type="date" bind:value={zeroShotStartDate} />
              </div>
              <div style="flex: 1; display: grid; gap: 6px;">
                <label for="zeroShotEndDate">End Date</label>
                <input id="zeroShotEndDate" type="date" bind:value={zeroShotEndDate} />
              </div>
            </div>

            <label for="zeroShotThreshold">Similarity Threshold</label>
            <input
              id="zeroShotThreshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={zeroShotThreshold}
            />

            <label for="zeroShotResolution">Resolution (m/px)</label>
            <input
              id="zeroShotResolution"
              type="number"
              min="10"
              max="60"
              step="10"
              bind:value={zeroShotResolution}
            />

            <button class="button-primary" on:click={handleZeroShotSearch} disabled={isLoading}>
              {isLoading ? 'Running...' : '🚀 Run Detection'}
            </button>
          </div>
        {:else}
          <div class="form-grid">
            <p class="status">
              Draw a query area (green) and a search area (orange) using the map.
            </p>

            <label for="copernicusSensor">Sensor</label>
            <select id="copernicusSensor" bind:value={copernicusSensor}>
              <option value="Sentinel-2">Sentinel-2</option>
              <option value="Sentinel-1">Sentinel-1</option>
            </select>

            <div class="form-actions">
              <div style="flex: 1; display: grid; gap: 6px;">
                <label for="copernicusStartDate">Start Date</label>
                <input id="copernicusStartDate" type="date" bind:value={copernicusStartDate} />
              </div>
              <div style="flex: 1; display: grid; gap: 6px;">
                <label for="copernicusEndDate">End Date</label>
                <input id="copernicusEndDate" type="date" bind:value={copernicusEndDate} />
              </div>
            </div>

            <label for="copernicusThreshold">Similarity Threshold</label>
            <input
              id="copernicusThreshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={copernicusThreshold}
            />

            <label for="copernicusResolution">Resolution (m/px)</label>
            <input
              id="copernicusResolution"
              type="number"
              min="10"
              max="60"
              step="10"
              bind:value={copernicusResolution}
            />

            <button class="button-primary" on:click={handleCopernicusSearch} disabled={isLoading}>
              {isLoading ? 'Searching...' : '🛰️ Run Copernicus'}
            </button>
          </div>
        {/if}

        {#if statusMessage}
          <p class="status">{statusMessage}</p>
        {/if}
      </div>
    </div>

    <section class="results">
      <h3>🎯 Search Results</h3>

      {#if results.length === 0}
        <p class="notice">Draw an area, enter a query, and click search to see results.</p>
      {:else}
        <div class="form-actions" style="margin-bottom: 16px; flex-wrap: wrap;">
          <button class="button-secondary" on:click={selectAllResults}>Select All</button>
          <button class="button-secondary" on:click={deselectAllResults}>Deselect All</button>
          <label style="display: flex; align-items: center; gap: 6px;">
            <input type="checkbox" bind:checked={showHeatmaps} />
            Show Heatmaps
          </label>
          {#if hasDinoAttention || hasPcaMap}
            <select bind:value={heatmapMode}>
              <option value="Similarity">Similarity</option>
              {#if hasDinoAttention}
                <option value="DINO Attention">DINO Attention</option>
              {/if}
              {#if hasPcaMap}
                <option value="PCA">PCA</option>
              {/if}
            </select>
          {/if}
          <button class="button-secondary" on:click={() => handleExport('pdf')}>PDF</button>
          <button class="button-secondary" on:click={() => handleExport('kmz')}>KMZ</button>
          <button class="button-secondary" on:click={() => handleExport('geojson')}>GeoJSON</button>
          <button class="button-secondary" on:click={() => handleExport('zip')}>ZIP</button>
        </div>

        <div class="results-grid">
          {#each results as result, index}
            <div class="result-card">
              <label style="display: flex; gap: 8px; align-items: center;">
                <input
                  type="checkbox"
                  checked={selectedResults.has(index)}
                  on:change={() => toggleResultSelection(index)}
                />
                Select
              </label>
              <div class="result-image">
                {#if result.image}
                  <img
                    src={`data:image/png;base64,${
                      showHeatmaps
                        ? heatmapMode === 'DINO Attention'
                          ? result.dino_attention || result.heatmap
                          : heatmapMode === 'PCA'
                            ? result.pca_map || result.heatmap
                            : result.heatmap
                        : result.image
                    }`}
                    alt="Result"
                  />
                {/if}
              </div>
              <div class="result-meta">
                <div>Score: {(result.score * 100).toFixed(1)}%</div>
                {#if result.bounds}
                  <div>
                    Bounds: {result.bounds[1].toFixed(4)}, {result.bounds[0].toFixed(4)}
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </section>
  </main>
</div>
