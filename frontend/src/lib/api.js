const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }

  return response;
}

export async function connectGEE(projectId) {
  const response = await request('/api/gee/connect', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId || null })
  });
  return response.json();
}

export async function semanticSearch(payload) {
  const response = await request('/api/search/semantic', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
  return response.json();
}

export async function zeroShotSearch(payload) {
  const response = await request('/api/search/zero-shot', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
  return response.json();
}

export async function copernicusSearch(payload) {
  const response = await request('/api/search/copernicus', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
  return response.json();
}

export async function exportResults(endpoint, payload) {
  const response = await request(`/api/export/${endpoint}`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
  return response.blob();
}
