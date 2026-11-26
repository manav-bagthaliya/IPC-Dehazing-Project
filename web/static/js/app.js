/* IPC-Dehaze front end. Plain JavaScript, no dependencies. User-supplied text is only ever set via textContent. */
(() => {
    'use strict';

    const $ = (sel, root = document) => root.querySelector(sel);
    const CFG = {
        csrf: document.body.dataset.csrf,
        maxImages: Number(document.body.dataset.maxImages) || 10,
        maxMb: Number(document.body.dataset.maxMb) || 100,
    };
    const PRESETS = {
        fast: { iterations: 4, max_size: 1024 },
        balanced: { iterations: 8, max_size: 1500 },
        best: { iterations: 12, max_size: 1500 },
    };
    const IMAGE_RE = /\.(png|jpe?g|bmp|tiff?|webp)$/i;
    const SETTINGS_KEY = 'ipc-settings';
    const LAST_JOB_KEY = 'ipc-last-job';

    let files = [];            // [{ id, file, url }]
    let job = null;            // latest job JSON from the server
    let busy = false;          // uploading or processing
    let pollTimer = null;
    const cardCache = new Map(); // item id -> { sig, el }

    // ---------------------------------------------------------------- helpers
    function h(tag, props = {}, ...children) {
        const el = document.createElement(tag);
        for (const [key, value] of Object.entries(props)) {
            if (value == null || value === false) continue;
            if (key === 'class') el.className = value;
            else if (key === 'text') el.textContent = value;
            else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
            else el.setAttribute(key, value === true ? '' : value);
        }
        for (const child of children.flat()) {
            if (child == null || child === false) continue;
            el.append(child.nodeType ? child : document.createTextNode(child));
        }
        return el;
    }
    const icon = (name) => {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'icon');
        const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
        use.setAttribute('href', '#i-' + name);
        svg.append(use);
        return svg;
    };
    const store = {
        get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
        set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be blocked */ } },
        del(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
    };
    const fmtMB = (bytes) => (bytes / 1048576).toFixed(bytes < 1048576 * 10 ? 1 : 0) + ' MB';
    const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

    function toast(message, bad = false) {
        const el = h('div', { class: 'toast' + (bad ? ' toast--bad' : ''), role: 'status' }, message);
        $('#toasts').append(el);
        setTimeout(() => el.remove(), bad ? 6500 : 3500);
    }

    async function api(path, options = {}) {
        const opts = { credentials: 'same-origin', ...options, headers: { ...(options.headers || {}) } };
        if (opts.method && opts.method !== 'GET') opts.headers['X-CSRF-Token'] = CFG.csrf;
        let response;
        try {
            response = await fetch(path, opts);
        } catch {
            throw new Error('Could not reach the server. Check your connection and try again.');
        }
        if (response.status === 401) { window.location.href = '/login'; throw new Error('Please sign in.'); }
        let data = null;
        try { data = await response.json(); } catch { /* not JSON */ }
        if (!response.ok) throw new Error((data && data.error) || `Request failed (${response.status}).`);
        return data;
    }

    // ---------------------------------------------------------------- engine status pill
    async function refreshEngine() {
        const pill = $('#enginePill'), text = $('#engineText');
        let next = 30000;
        try {
            const res = await fetch('/health', { cache: 'no-store' });
            const data = await res.json();
            const e = data.engine || {};
            pill.className = 'pill';
            if (data.status === 'error') { pill.classList.add('pill--bad'); text.textContent = 'Engine error'; pill.title = e.error || ''; }
            else if (!e.ready) { pill.classList.add('pill--warn'); text.textContent = 'Loading models…'; next = 2000; }
            else if (e.gpu_name && e.device.startsWith('cuda')) {
                pill.classList.add('pill--ok');
                text.textContent = 'GPU · ' + e.gpu_name.replace(/^NVIDIA GeForce /, '').replace(/ Laptop GPU$/, '');
            } else { pill.classList.add('pill--warn'); text.textContent = 'CPU mode (slower)'; }
        } catch {
            pill.className = 'pill pill--bad'; text.textContent = 'Offline'; next = 5000;
        }
        setTimeout(refreshEngine, next);
    }

    // ---------------------------------------------------------------- settings
    const fields = {
        iter: $('#optIter'), max: $('#optMax'), resize: $('#optResize'),
        fw: $('#optFW'), fh: $('#optFH'), post: $('#optPost'), plates: $('#optPlates'),
    };
    const presetRadios = [...document.querySelectorAll('input[name=preset]')];
    const currentPreset = () => (presetRadios.find((r) => r.checked) || {}).value || 'balanced';

    function applyPreset(name) {
        const p = PRESETS[name];
        if (!p) return;
        fields.iter.value = p.iterations;
        fields.max.value = p.max_size;
    }
    function readParams() {
        const num = (el, d) => (Number.isFinite(el.valueAsNumber) ? el.valueAsNumber : d);
        return {
            iterations: num(fields.iter, 8), max_size: num(fields.max, 1500), resize_to: num(fields.resize, 0),
            force_width: num(fields.fw, 0), force_height: num(fields.fh, 0),
            detect_plates: fields.plates.checked, post_enhance: fields.post.checked,
        };
    }
    function saveSettings() {
        store.set(SETTINGS_KEY, { preset: currentPreset(), ...readParams() });
    }
    function loadSettings() {
        const s = store.get(SETTINGS_KEY);
        if (!s) return;
        const radio = presetRadios.find((r) => r.value === s.preset);
        if (radio) radio.checked = true;
        fields.iter.value = s.iterations ?? 8; fields.max.value = s.max_size ?? 1500;
        fields.resize.value = s.resize_to ?? 0; fields.fw.value = s.force_width ?? 0; fields.fh.value = s.force_height ?? 0;
        fields.plates.checked = s.detect_plates !== false; fields.post.checked = !!s.post_enhance;
    }

    // ---------------------------------------------------------------- selecting files
    const dropzone = $('#dropzone'), fileInput = $('#fileInput');

    function addFiles(list) {
        let skipped = 0, tooMany = false;
        for (const file of list) {
            if (!(file.type.startsWith('image/') || IMAGE_RE.test(file.name)) || !IMAGE_RE.test(file.name || 'x.png')) { skipped++; continue; }
            if (files.length >= CFG.maxImages) { tooMany = true; break; }
            files.push({ id: crypto.randomUUID ? crypto.randomUUID() : String(Math.random()), file, url: URL.createObjectURL(file) });
        }
        const total = files.reduce((n, f) => n + f.file.size, 0);
        if (total > CFG.maxMb * 1048576) {
            while (files.length && files.reduce((n, f) => n + f.file.size, 0) > CFG.maxMb * 1048576) {
                URL.revokeObjectURL(files.pop().url);
            }
            toast(`That is more than ${CFG.maxMb} MB in total. Some images were left out.`, true);
        }
        if (skipped) toast(`${plural(skipped, 'file')} skipped: only PNG, JPG, BMP, TIFF and WebP images work.`, true);
        if (tooMany) toast(`You can add up to ${CFG.maxImages} images at a time.`, true);
        renderThumbs();
    }
    function removeFile(id) {
        const i = files.findIndex((f) => f.id === id);
        if (i >= 0) { URL.revokeObjectURL(files[i].url); files.splice(i, 1); renderThumbs(); }
    }
    function clearFiles() {
        files.forEach((f) => URL.revokeObjectURL(f.url));
        files = [];
        renderThumbs();
    }
    function renderThumbs() {
        const list = $('#thumbs');
        list.replaceChildren(...files.map((f) => h('li', { class: 'thumb' },
            h('img', { src: f.url, alt: '', loading: 'lazy' }),
            h('div', { class: 'thumb__meta', title: f.file.name }, `${f.file.name} · ${fmtMB(f.file.size)}`),
            h('button', { class: 'thumb__remove', type: 'button', 'aria-label': `Remove ${f.file.name}`, onclick: () => removeFile(f.id) },
                icon('x')))));
        $('#clearBtn').hidden = files.length === 0;
        updateProcessButton();
    }
    function updateProcessButton() {
        const btn = $('#processBtn'), label = $('#processLabel');
        btn.disabled = busy || files.length === 0;
        label.textContent = busy ? 'Working…' : files.length ? `Clean ${plural(files.length, 'image')}` : 'Add images to start';
    }

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
    fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });
    ['dragenter', 'dragover'].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove('is-over'); }));
    dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
    // Window-level drop so a missed drop never navigates away from the page
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => { e.preventDefault(); if (!dropzone.contains(e.target) && e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });
    document.addEventListener('paste', (e) => {
        const pasted = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith('image/'))
            .map((f) => (f.name && IMAGE_RE.test(f.name) ? f : new File([f], 'pasted-' + Date.now() + '.png', { type: f.type })));
        if (pasted.length) addFiles(pasted);
    });
    $('#clearBtn').addEventListener('click', clearFiles);

    presetRadios.forEach((r) => r.addEventListener('change', () => { applyPreset(r.value); saveSettings(); }));
    Object.values(fields).forEach((el) => el.addEventListener('change', saveSettings));

    // ---------------------------------------------------------------- running a job
    async function startJob() {
        if (busy || !files.length) return;
        const error = $('#formError');
        error.hidden = true;
        busy = true; updateProcessButton();
        saveSettings();

        const form = new FormData();
        files.forEach((f) => form.append('files', f.file, f.file.name));
        form.append('params', JSON.stringify(readParams()));
        try {
            job = await api('/api/jobs', { method: 'POST', body: form });
            store.set(LAST_JOB_KEY, job.id);
            history.replaceState(null, '', '?job=' + job.id);
            clearFiles();
            showResults();
            schedulePoll(600);
        } catch (err) {
            error.textContent = err.message;
            error.hidden = false;
            busy = false; updateProcessButton();
        }
    }
    $('#processBtn').addEventListener('click', startJob);

    function schedulePoll(delay = 1000, failures = 0) {
        clearTimeout(pollTimer);
        pollTimer = setTimeout(async () => {
            if (!job) return;
            try {
                job = await api('/api/jobs/' + job.id);
                showResults();
                if (job.status === 'queued' || job.status === 'running') schedulePoll(1000);
                else { busy = false; updateProcessButton(); }
            } catch (err) {
                if (failures >= 4) { toast(err.message, true); busy = false; updateProcessButton(); }
                else schedulePoll(2000 * (failures + 1), failures + 1);
            }
        }, delay);
    }

    async function cancelJob() {
        try { job = await api(`/api/jobs/${job.id}/cancel`, { method: 'POST' }); showResults(); toast('Stopping after the current image…'); }
        catch (err) { toast(err.message, true); }
    }

    function newBatch() {
        clearTimeout(pollTimer);
        job = null; busy = false;
        cardCache.clear();
        $('#resultsSection').hidden = true;
        store.del(LAST_JOB_KEY);
        history.replaceState(null, '', location.pathname);
        updateProcessButton();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // ---------------------------------------------------------------- rendering results
    const TITLES = {
        queued: 'Waiting in line…', running: 'Cleaning your images…', done: 'All done',
        error: 'Something went wrong', cancelled: 'Cancelled',
    };

    function showResults() {
        if (!job) return;
        const section = $('#resultsSection');
        section.hidden = false;
        const { done, total, percent } = job.progress;
        const active = job.status === 'queued' || job.status === 'running';

        $('#resultsTitle').textContent = TITLES[job.status] || 'Results';
        $('#resultsSub').textContent = active ? `${done} of ${plural(total, 'image')} finished` : (job.error || '');
        const bar = $('#progress');
        bar.classList.toggle('is-done', !active);
        bar.setAttribute('aria-valuenow', String(percent));
        $('#progressBar').style.width = Math.max(percent, active ? 4 : 100) + '%';

        const actions = $('#resultsActions');
        actions.replaceChildren();
        if (active) actions.append(h('button', { class: 'btn', type: 'button', onclick: cancelJob }, 'Cancel'));
        else {
            if (job.downloads) {
                actions.append(h('a', { class: 'btn btn--primary', href: job.downloads.zip }, icon('download'), 'Download all (ZIP)'));
                if (job.summary.plates_found) actions.append(h('a', { class: 'btn', href: job.downloads.csv + '?download=1' }, icon('file'), 'Plates (CSV)'));
            }
            actions.append(h('button', { class: 'btn', type: 'button', onclick: newBatch }, icon('plus'), 'New batch'));
        }
        renderSummary();
        renderCards();
        busy = active;
        updateProcessButton();
    }

    function renderSummary() {
        const box = $('#summary');
        box.replaceChildren();
        if (job.status !== 'done' && job.status !== 'cancelled') return;
        const items = job.items;
        const ok = items.filter((i) => i.status === 'done').length;
        const failed = items.filter((i) => i.status === 'error').length;
        const secs = items.reduce((n, i) => n + (i.seconds || 0), 0);
        box.append(h('span', { class: 'chip chip--ok' }, icon('check'), `${plural(ok, 'image')} cleaned`));
        if (failed) box.append(h('span', { class: 'chip chip--bad' }, `${failed} failed`));
        if (job.params.detect_plates) {
            box.append(h('span', { class: 'chip' }, `${plural(job.summary.plates_found, 'plate')} found`));
            if (job.summary.plates_found) box.append(h('span', { class: job.summary.plates_valid ? 'chip chip--ok' : 'chip chip--warn' },
                `${job.summary.plates_valid} read as valid Indian plates`));
        }
        if (secs) box.append(h('span', { class: 'chip' }, `${secs.toFixed(1)} s total`));
    }

    function renderCards() {
        const container = $('#cards');
        const seen = new Set();
        const order = [];
        for (const item of job.items) {
            seen.add(item.id);
            const sig = JSON.stringify(item);
            let entry = cardCache.get(item.id);
            if (!entry || entry.sig !== sig) {
                entry = { sig, el: buildCard(item) };
                cardCache.set(item.id, entry);
            }
            order.push(entry.el);
        }
        for (const id of [...cardCache.keys()]) if (!seen.has(id)) cardCache.delete(id);
        container.replaceChildren(...order);
    }

    function buildCard(item) {
        const meta = item.status === 'done'
            ? [item.width && `${item.width}×${item.height}`, item.seconds != null && `${item.seconds.toFixed(1)} s`].filter(Boolean).join(' · ') : '';
        const card = h('article', { class: 'result' },
            h('div', { class: 'result__head' }, h('span', { class: 'result__name', title: item.name }, item.name),
                h('span', { class: 'result__meta' }, meta)));

        if (item.status === 'queued') card.append(h('div', { class: 'result__state' }, h('span', {}, 'Waiting…')));
        else if (item.status === 'processing') card.append(h('div', { class: 'result__state' }, h('div', { class: 'spinner' }), h('span', {}, 'Cleaning the haze…')));
        else if (item.status === 'cancelled') card.append(h('div', { class: 'result__state' }, h('span', {}, 'Skipped')));
        else if (item.status === 'error') card.append(h('div', { class: 'result__error', role: 'alert' }, icon('alert'), item.error || 'This image could not be processed.'));
        else buildDone(card, item);
        return card;
    }

    function buildDone(card, item) {
        const f = item.files;
        const hasPlates = item.plates.length > 0;
        const views = [
            { id: 'compare', label: 'Before / After' },
            { id: 'dehazed', label: 'Cleaned', src: f.dehazed },
            ...(hasPlates ? [{ id: 'result', label: 'With plates', src: f.result }] : []),
        ];
        const stage = h('div', { class: 'stage' });
        const tabs = h('div', { class: 'tabs', role: 'tablist' });

        function show(view) {
            tabs.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.view === view.id)));
            stage.replaceChildren(view.id === 'compare' ? buildCompare(item) : zoomable(h('img', { src: view.src, alt: `${item.name}: ${view.label}` })));
        }
        views.forEach((v, i) => tabs.append(h('button', { class: 'tab', type: 'button', role: 'tab', 'data-view': v.id, onclick: () => show(v) }, v.label)));
        card.append(tabs, stage);
        show(views[0]);

        const body = h('div', { class: 'result__body' });
        if (item.fallback) body.append(h('div', { class: 'chip chip--warn' }, 'Result looked blank, so the original is shown'));
        if (hasPlates) {
            body.append(h('div', { class: 'plates' }, h('div', { class: 'plates__title' }, 'Licence plates'), item.plates.map(plateRow)));
        } else if (job.params.detect_plates) {
            body.append(h('p', { class: 'muted' }, 'No licence plates found in this image.'));
        }
        body.append(h('div', { class: 'result__links' },
            h('a', { class: 'btn btn--small', href: f.dehazed_png + '?download=1' }, icon('download'), 'Cleaned (PNG)'),
            hasPlates ? h('a', { class: 'btn btn--small', href: f.result_png + '?download=1' }, icon('download'), 'With plates (PNG)') : null));
        card.append(body);
    }

    function buildCompare(item) {
        const f = item.files;
        const range = h('input', { class: 'compare__range', type: 'range', min: '0', max: '100', value: '50', 'aria-label': 'Drag to compare before and after' });
        const wrap = h('div', { class: 'compare' },
            h('div', { class: 'compare__after' }, h('img', { src: f.dehazed, alt: `${item.name}: after`, draggable: 'false' })),
            h('div', { class: 'compare__before' }, h('img', { src: f.original, alt: `${item.name}: before`, draggable: 'false' })),
            range, h('div', { class: 'compare__line' }),
            h('span', { class: 'tag tag--l' }, 'Before'), h('span', { class: 'tag tag--r' }, 'After'));
        wrap.style.setProperty('--pos', '50%');
        range.addEventListener('input', () => wrap.style.setProperty('--pos', range.value + '%'));
        return wrap;
    }

    function zoomable(img) { img.addEventListener('click', () => openLightbox(img.src)); return img; }

    function plateRow(plate) {
        const text = plate.formatted_text || plate.text || '';
        const badge = plate.valid ? h('span', { class: 'badge badge--ok' }, 'Valid Indian plate')
            : h('span', { class: 'badge badge--warn' }, text ? 'Check by eye' : 'Could not read');
        const crop = plate.crop ? zoomable(h('img', { src: plate.crop, alt: 'Plate close-up', loading: 'lazy' })) : null;
        const copy = text ? h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Copy plate number', title: 'Copy',
            onclick: async () => { try { await navigator.clipboard.writeText(text); toast('Copied ' + text); } catch { toast('Could not copy', true); } } }, icon('copy')) : null;
        return h('div', { class: 'plate' }, crop,
            h('div', { class: 'plate__main' },
                h('span', { class: 'plate__text' }, text || '—'),
                h('span', { class: 'plate__sub' }, `${plate.color && plate.color !== 'unknown' ? plate.color + ' plate · ' : ''}found by ${plate.method === 'yolo' ? 'plate detector' : plate.method}`)),
            badge, copy);
    }

    // ---------------------------------------------------------------- lightbox, drawer, menus
    const lightbox = $('#lightbox');
    function openLightbox(src) { $('#lightboxImg').src = src; lightbox.hidden = false; }
    function closeLightbox() { lightbox.hidden = true; $('#lightboxImg').removeAttribute('src'); }
    lightbox.addEventListener('click', (e) => { if (e.target !== $('#lightboxImg')) closeLightbox(); });

    const drawer = $('#drawer');
    async function openHistory() {
        drawer.hidden = false;
        const list = $('#historyList');
        list.replaceChildren(h('li', { class: 'empty' }, 'Loading…'));
        try {
            const { jobs } = await api('/api/jobs');
            list.replaceChildren(...(jobs.length ? jobs.map(historyRow) : [h('li', { class: 'empty' }, 'No batches yet.')]));
        } catch (err) { list.replaceChildren(h('li', { class: 'empty' }, err.message)); }
        $('#drawerClose').focus();
    }
    function historyRow(j) {
        const when = new Date(j.created * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
        const state = j.status === 'done' ? '' : ` · ${j.status}`;
        return h('li', { class: 'history__item' },
            j.thumb ? h('img', { class: 'history__thumb', src: j.thumb, alt: '', loading: 'lazy' }) : h('div', { class: 'history__thumb' }),
            h('button', { class: 'history__open', type: 'button', onclick: () => openJob(j.id) },
                h('strong', {}, plural(j.images, 'image')), h('span', {}, `${when}${state}${j.plates ? ' · ' + plural(j.plates, 'plate') : ''}`)),
            h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Delete this batch', title: 'Delete', onclick: () => removeJob(j.id) }, icon('trash')));
    }
    async function openJob(id) {
        try {
            clearTimeout(pollTimer);
            cardCache.clear();
            job = await api('/api/jobs/' + id);
            store.set(LAST_JOB_KEY, id);
            history.replaceState(null, '', '?job=' + id);
            drawer.hidden = true;
            showResults();
            if (job.status === 'queued' || job.status === 'running') schedulePoll(800);
            $('#resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (err) { toast(err.message, true); }
    }
    async function removeJob(id) {
        if (!confirm('Delete this batch and its results?')) return;
        try {
            await api('/api/jobs/' + id, { method: 'DELETE' });
            if (job && job.id === id) newBatch();
            openHistory();
        } catch (err) { toast(err.message, true); }
    }
    $('#historyBtn').addEventListener('click', openHistory);
    $('#drawerClose').addEventListener('click', () => { drawer.hidden = true; });
    $('#drawerScrim').addEventListener('click', () => { drawer.hidden = true; });

    const userBtn = $('#userBtn'), userMenu = $('#userDropdown');
    if (userBtn) {
        const toggle = (open) => { userMenu.hidden = !open; userBtn.setAttribute('aria-expanded', String(open)); };
        userBtn.addEventListener('click', (e) => { e.stopPropagation(); toggle(userMenu.hidden); });
        document.addEventListener('click', (e) => { if (!userMenu.contains(e.target)) toggle(false); });
        $('#logoutBtn').addEventListener('click', async () => {
            try { await api('/logout', { method: 'POST' }); } catch { /* leave anyway */ }
            store.del(LAST_JOB_KEY);
            window.location.href = '/login';
        });
    }
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        closeLightbox(); drawer.hidden = true;
        if (userBtn) { userMenu.hidden = true; userBtn.setAttribute('aria-expanded', 'false'); }
    });

    // ---------------------------------------------------------------- start up
    loadSettings();
    renderThumbs();
    refreshEngine();

    const wanted = new URLSearchParams(location.search).get('job') || store.get(LAST_JOB_KEY);
    if (wanted && /^[a-f0-9]{8,32}$/.test(wanted)) {
        api('/api/jobs/' + wanted).then((data) => {
            job = data;
            showResults();
            if (job.status === 'queued' || job.status === 'running') schedulePoll(800);
        }).catch(() => { store.del(LAST_JOB_KEY); history.replaceState(null, '', location.pathname); });
    }
})();
