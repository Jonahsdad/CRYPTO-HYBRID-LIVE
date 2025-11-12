// worker/cloudflare/worker.js
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // only cache API v1 calls; everything else goes straight to origin
    if (!url.pathname.startsWith("/api/v1/")) {
      return fetch(request);
    }

    const cacheKey = new Request(url.toString(), request);
    const cache = caches.default;

    // try cache first
    let res = await cache.match(cacheKey);
    if (res) {
      return res;
    }

    // hit origin
    res = await fetch(request);

    // cautious TTLs: shorter for /predict
    const ttl = url.pathname.includes("/predict/") ? 15 : 60;
    const cached = new Response(res.body, res);
    cached.headers.set("Cache-Control", `public, max-age=${ttl}`);

    ctx.waitUntil(cache.put(cacheKey, cached.clone()));
    return cached;
  }
};
