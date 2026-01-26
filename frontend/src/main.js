import App from './App.svelte';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import './styles.css';

const app = new App({
  target: document.getElementById('app')
});

export default app;
