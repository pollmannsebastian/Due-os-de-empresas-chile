/* ═══════════════════════════════════════════════════════════════════════════
   De Quiénes — Graph Explorer
   Expandable network, draggable nodes, source documents
   ═══════════════════════════════════════════════════════════════════════════ */

let network = null;
let nodesDataset = null;
let edgesDataset = null;
let allGraphData = { nodes: {}, edges: {} }; // Accumulated graph state
let searchTimeout = null;

const COLORS = {
    person: { bg: '#4a9eff', border: '#3a7ecc', highlight: '#6ab4ff' },
    company: { bg: '#8b5cf6', border: '#6d3ed4', highlight: '#a87dff' },
};

// ── Search ─────────────────────────────────────────────────────────────────
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');
const consolidateToggle = document.getElementById('consolidateToggle');

searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    const q = searchInput.value.trim();
    if (q.length < 2) { searchResults.classList.add('hidden'); return; }
    searchTimeout = setTimeout(() => doSearch(q), 250);
});

searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { searchResults.classList.add('hidden'); searchInput.blur(); }
});

document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-section')) searchResults.classList.add('hidden');
});

async function doSearch(q) {
    try {
        const cons = consolidateToggle.checked;
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&consolidate=${cons}`);
        const data = await res.json();
        renderSearchResults(data);
    } catch (err) { console.error('Search error:', err); }
}

function renderSearchResults(results) {
    if (!results.length) {
        searchResults.innerHTML = '<div style="padding:14px;color:#555570;text-align:center;font-size:12px;">Sin resultados</div>';
        searchResults.classList.remove('hidden');
        return;
    }
    searchResults.innerHTML = results.map(r => `
        <div class="result-item" onclick="loadNode('${r.id}', true)">
            <span class="result-type-badge ${r.type}">${r.type === 'person' ? '👤' : '🏢'}</span>
            <div class="result-info">
                <div class="result-name">
                    ${esc(r.name)}
                    ${r.count > 1 ? `<span class="result-count">(${r.count} docs)</span>` : ''}
                </div>
                ${r.rut ? `<div class="result-rut">${fmtRut(r.rut)}</div>` : ''}
            </div>
        </div>
    `).join('');
    searchResults.classList.remove('hidden');
}

// ── Graph ──────────────────────────────────────────────────────────────────
let currentCenterId = null; // Track who we are looking at to allow refresh on toggle

consolidateToggle.addEventListener('change', () => {
    if (currentCenterId) {
        loadNode(currentCenterId, true);
    }
});

async function loadNode(nodeId, resetGraph = false) {
    currentCenterId = nodeId;
    searchResults.classList.add('hidden');
    showLoading(true);

    try {
        const cons = consolidateToggle.checked;
        const res = await fetch(`/api/graph/${nodeId}?consolidate=${cons}`);
        const data = await res.json();
        if (data.error) { showLoading(false); return; }

        if (resetGraph) {
            allGraphData = { nodes: {}, edges: {} };
        }

        mergeGraphData(data);
        renderGraph(nodeId);
        showNodeDetail(data.center, data);
        updateStats();
    } catch (err) { console.error('Graph error:', err); }
    showLoading(false);
}

async function expandNode(nodeId) {
    showLoading(true);
    try {
        const cons = consolidateToggle.checked;
        const res = await fetch(`/api/expand/${nodeId}?consolidate=${cons}`);
        const data = await res.json();
        if (data.error) { showLoading(false); return; }

        mergeGraphData(data);
        renderGraph(null);
        showNodeDetail(data.center, data);
        updateStats();
    } catch (err) { console.error('Expand error:', err); }
    showLoading(false);
}

function normalizeName(name) {
    if (!name) return "";
    return String(name).toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]/g, '');
}

function cleanCompanyName(name) {
    if (!name) return "";
    let s = String(name).toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    s = s.replace(/[^\w\s]/g, ' ');
    const stops = ["spa", "limitada", "ltda", "sa", "s a", "eirl", "sociedad", "anonima", "inversiones", "comercial", "constructora", "y", "cia", "empresas", "grupo", "los", "las", "el", "la", "de"];
    let words = s.split(/\s+/).filter(Boolean);
    let core = words.filter(w => !stops.includes(w));
    return core.length > 0 ? core.join('') : words.join('');
}

function similarity(s1, s2) {
    let longer = s1, shorter = s2;
    if (s1.length < s2.length) { longer = s2; shorter = s1; }
    if (longer.length == 0) return 1.0;
    return (longer.length - editDistance(longer, shorter)) / parseFloat(longer.length);
}

function editDistance(s1, s2) {
    let costs = [];
    for (let i = 0; i <= s1.length; i++) {
        let lastValue = i;
        for (let j = 0; j <= s2.length; j++) {
            if (i == 0) costs[j] = j;
            else if (j > 0) {
                let newValue = costs[j - 1];
                if (s1.charAt(i - 1) !== s2.charAt(j - 1))
                    newValue = Math.min(Math.min(newValue, lastValue), costs[j]) + 1;
                costs[j - 1] = lastValue;
                lastValue = newValue;
            }
        }
        if (i > 0) costs[s2.length] = lastValue;
    }
    return costs[s2.length];
}

function mergeGraphData(data) {
    const isConsolidated = consolidateToggle.checked;

    const remappedIds = {};
    const resolveId = (id) => {
        let curr = id;
        let depth = 0;
        while (remappedIds[curr] && remappedIds[curr] !== curr && depth < 10) {
            curr = remappedIds[curr];
            depth++;
        }
        return curr;
    };

    for (const n of data.nodes) {
        if (isConsolidated) {
            const newIds = String(n.id).split(',');
            let matchedExistingIds = [];

            const currentExistingIds = Object.keys(allGraphData.nodes);
            for (const existingId of currentExistingIds) {
                const exNode = allGraphData.nodes[existingId];
                const exIds = String(existingId).split(',');
                let sameIdentity = false;

                // 1. Exact ID overlap (Backend already merged them)
                if (exIds.some(id => newIds.includes(id))) {
                    sameIdentity = true;
                }
                // 2. Identity match fallback
                else if (n.type === exNode.type) {
                    if (n.type === 'person') {
                        if (n.rut && exNode.rut && n.rut === exNode.rut) sameIdentity = true;
                        else if (normalizeName(n.name) === normalizeName(exNode.name)) sameIdentity = true;
                    } else {
                        // Companies ONLY merge if they share a valid RUT OR fuzzy-match
                        if (n.rut && exNode.rut && n.rut === exNode.rut) {
                            sameIdentity = true;
                        } 
                        // Cross-check for fuzzy name in case they were fetched separately
                        else if (!n.rut || !exNode.rut) {
                           let c1 = cleanCompanyName(n.name);
                           let c2 = cleanCompanyName(exNode.name);
                           if (c1 && c2) {
                               if (c1 === c2) {
                                   sameIdentity = true;
                               } else if (Math.abs(c1.length - c2.length) <= 15) {
                                   if (similarity(c1, c2) >= 0.85) {
                                       sameIdentity = true;
                                   }
                               }
                           }
                        }
                    }
                }

                if (sameIdentity) {
                    matchedExistingIds.push(existingId);
                }
            }

            if (matchedExistingIds.length > 0) {
                let combinedIds = new Set(newIds);
                let combinedNames = new Set(n.names || []);
                let bestRut = n.rut;

                for (const exId of matchedExistingIds) {
                    const exNode = allGraphData.nodes[exId];
                    String(exId).split(',').forEach(id => combinedIds.add(id));
                    (exNode.names || []).forEach(name => combinedNames.add(name));
                    if (!bestRut && exNode.rut) bestRut = exNode.rut;
                }

                const combinedId = [...combinedIds].sort().join(',');
                const mergedNode = {
                    ...allGraphData.nodes[matchedExistingIds[0]], // fallback to first match
                    ...n, // apply new
                    id: combinedId,
                    rut: bestRut,
                    names: [...combinedNames].sort(),
                };

                for (const exId of matchedExistingIds) {
                    delete allGraphData.nodes[exId];
                    remappedIds[exId] = combinedId;
                }
                remappedIds[n.id] = combinedId;
                allGraphData.nodes[combinedId] = mergedNode;
                continue;
            }
        }
        allGraphData.nodes[n.id] = n;
        remappedIds[n.id] = n.id;
    }

    // Now remap all edges that pointed to the old IDs
    for (const eKey of Object.keys(allGraphData.edges)) {
        const e = allGraphData.edges[eKey];
        // resolveId safely walks the chain of remappedIds to the final ID
        let src = e.source;
        while(remappedIds[src] && remappedIds[src] !== src) src = remappedIds[src];
        
        let tgt = e.target;
        while(remappedIds[tgt] && remappedIds[tgt] !== tgt) tgt = remappedIds[tgt];

        if (src !== e.source || tgt !== e.target) {
            delete allGraphData.edges[eKey];
            if (src === tgt) continue; // prevent self-loops
            
            const newKey = `${src}-${tgt}-${e.relation}`;
            e.source = src;
            e.target = tgt;
            
            if (allGraphData.edges[newKey]) {
                allGraphData.edges[newKey].count = Math.max(allGraphData.edges[newKey].count, e.count);
                const existingUrls = new Set((allGraphData.edges[newKey].documents || []).map(d => d.url));
                for (const doc of (e.documents || [])) {
                    if (!existingUrls.has(doc.url)) {
                        allGraphData.edges[newKey].documents = allGraphData.edges[newKey].documents || [];
                        allGraphData.edges[newKey].documents.push(doc);
                    }
                }
            } else {
                allGraphData.edges[newKey] = e;
            }
        }
    }

    for (const e of data.edges) {
        const source = resolveId(e.source);
        const target = resolveId(e.target);

        if (source === target) continue;

        const key = `${source}-${target}-${e.relation}`;
        if (!allGraphData.edges[key]) {
            allGraphData.edges[key] = { ...e, source, target };
        } else {
            const existing = allGraphData.edges[key];
            existing.count = Math.max(existing.count, e.count);
            if (e.documents) {
                const existingUrls = new Set((existing.documents || []).map(d => d.url));
                for (const doc of e.documents) {
                    if (!existingUrls.has(doc.url)) {
                        existing.documents = existing.documents || [];
                        existing.documents.push(doc);
                    }
                }
            }
        }
    }
}

function renderGraph(centerId) {
    const ph = document.getElementById('graphPlaceholder');
    if (ph) ph.style.display = 'none';

    const nodes = Object.values(allGraphData.nodes).map(n => ({
        id: n.id,
        label: buildLabel(n),
        shape: n.type === 'company' ? 'diamond' : 'dot',
        size: 20,
        color: {
            background: COLORS[n.type].bg,
            border: COLORS[n.type].border,
            highlight: { background: COLORS[n.type].highlight, border: COLORS[n.type].highlight },
        },
        font: { color: '#c8c8d8', size: 11, face: 'Inter, sans-serif' },
        borderWidth: 1.5,
        nodeData: n,
    }));

    // Store edge key on each vis edge so we can look up allGraphData.edges on click
    const edges = Object.entries(allGraphData.edges).map(([key, e], i) => {
        const edgeObj = {
            id: `e${i}`,
            from: e.source,
            to: e.target,
            color: { color: 'rgba(100,100,140,0.3)', highlight: 'rgba(74,158,255,0.6)' },
            width: Math.min(1 + Math.log2(e.count || 1), 3.5),
            smooth: { type: 'continuous' },
            arrows: { to: { enabled: true, scaleFactor: 0.4 } },
            font: { color: '#8b8ba0', size: 10, align: 'top', strokeWidth: 0 },
            edgeKey: key,
        };
        // Add percentage label if it exists
        if (e.percentage !== undefined && e.percentage !== null) {
            edgeObj.label = `${e.percentage}%`;
        }
        return edgeObj;
    });

    nodesDataset = new vis.DataSet(nodes);
    edgesDataset = new vis.DataSet(edges);

    const container = document.getElementById('graphContainer');

    const options = {
        nodes: {
            font: { multi: false },
            fixed: { x: false, y: false },
        },
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -35,
                centralGravity: 0.006,
                springLength: 130,
                springConstant: 0.035,
                damping: 0.5,
            },
            stabilization: { iterations: 120, fit: true },
        },
        interaction: {
            hover: true,
            dragNodes: true,
            zoomSpeed: 0.8,
        },
    };

    if (network) network.destroy();
    network = new vis.Network(container, { nodes: nodesDataset, edges: edgesDataset }, options);

    // After stabilization, turn off physics so nodes are freely draggable
    network.once('stabilizationIterationsDone', () => {
        network.setOptions({ physics: { enabled: false } });
    });

    // Click → node detail OR edge detail
    network.on('click', (params) => {
        if (params.nodes.length > 0) {
            // Node clicked
            const clickedId = params.nodes[0];
            const node = nodesDataset.get(clickedId);
            if (node && node.nodeData) {
                showNodeDetail(node.nodeData, null);
            }
        } else if (params.edges.length > 0) {
            // Edge clicked — no node selected
            const visEdge = edgesDataset.get(params.edges[0]);
            if (visEdge && visEdge.edgeKey) {
                const edgeData = allGraphData.edges[visEdge.edgeKey];
                if (edgeData) showEdgeDetail(edgeData);
            }
        }
    });

    // Double-click → expand connections
    network.on('doubleClick', (params) => {
        if (params.nodes.length > 0) {
            expandNode(params.nodes[0]);
        }
    });

    // Focus on center
    if (centerId) {
        network.once('stabilizationIterationsDone', () => {
            network.focus(centerId, { scale: 1.0, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        });
    }
}

function buildLabel(node) {
    let label = capitalize(node.name);
    if (label.length > 30) label = label.substring(0, 27) + '...';
    if (node.rut) label += `\n${fmtRut(node.rut)}`;
    return label;
}

// ── Edge Detail Panel ──────────────────────────────────────────────────────
function showEdgeDetail(edge) {
    const section = document.getElementById('detailSection');
    const content = document.getElementById('detailContent');
    const instructions = document.getElementById('instructions');

    instructions.style.display = 'none';
    section.classList.remove('hidden');

    const srcNode = allGraphData.nodes[edge.source];
    const tgtNode = allGraphData.nodes[edge.target];
    const srcName = srcNode ? capitalize(srcNode.name) : `#${edge.source}`;
    const tgtName = tgtNode ? capitalize(tgtNode.name) : `#${edge.target}`;

    const docs = (edge.documents || []).filter(d => d.url);
    const seen = new Set();
    const uniqueDocs = docs.filter(d => {
        if (seen.has(d.url)) return false;
        seen.add(d.url);
        return true;
    });

    let html = `
        <div class="detail-header">
            <div class="detail-name" style="font-size:13px;line-height:1.5;">
                ${esc(srcName)}
                <span style="color:#555570;font-weight:400;"> — </span>
                ${esc(tgtName)}
            </div>
            ${edge.relation ? `<div class="detail-rut">${esc(edge.relation)}</div>` : ''}
            <span class="detail-type company">🔗 Conexión · ${edge.count} doc${edge.count !== 1 ? 's' : ''}</span>
            ${edge.percentage ? `<div style="margin-top:8px; font-size:14px; font-weight:bold; color:#a87dff;">Participación: ${edge.percentage}%</div>` : ''}
        </div>
    `;

    if (uniqueDocs.length > 0) {
        html += `
            <div class="detail-block">
                <h3>📄 Documentos que originan esta conexión</h3>
                ${uniqueDocs.map(d => `
                    <a class="doc-link" href="${d.url}" target="_blank" title="${d.url}">
                        <span class="doc-date">${d.date || ''}</span>CVE ${d.cve}
                    </a>
                `).join('')}
            </div>
        `;
    } else {
        html += `
            <div class="detail-block">
                <p style="color:#555570;font-size:12px;">No hay documentos disponibles para esta conexión.</p>
            </div>
        `;
    }

    // Quick links to expand either endpoint
    html += `
        <div class="detail-block">
            <h3>🔍 Explorar</h3>
            ${srcNode ? `<div class="connection-item" onclick="expandNode(${srcNode.id})">
                <span class="connection-dot ${srcNode.type}"></span>
                <span class="connection-name">${esc(capitalize(srcNode.name))}</span>
            </div>` : ''}
            ${tgtNode ? `<div class="connection-item" onclick="expandNode(${tgtNode.id})">
                <span class="connection-dot ${tgtNode.type}"></span>
                <span class="connection-name">${esc(capitalize(tgtNode.name))}</span>
            </div>` : ''}
        </div>
    `;

    content.innerHTML = html;
}

// ── Node Detail Panel ──────────────────────────────────────────────────────
async function showNodeDetail(node, graphData) {
    const section = document.getElementById('detailSection');
    const content = document.getElementById('detailContent');
    const instructions = document.getElementById('instructions');

    instructions.style.display = 'none';
    section.classList.remove('hidden');

    const connections = [];
    const documents = [];
    for (const e of Object.values(allGraphData.edges)) {
        let connectedId = null;
        if (e.source === node.id) connectedId = e.target;
        else if (e.target === node.id) connectedId = e.source;
        if (connectedId !== null) {
            const connNode = allGraphData.nodes[connectedId];
            if (connNode) {
                connections.push({ node: connNode, count: e.count });
            }
            if (e.documents) {
                documents.push(...e.documents);
            }
        }
    }

    let html = `
        <div class="detail-header">
            <div class="detail-name">${esc(node.name)}</div>
            ${node.rut ? `<div class="detail-rut">RUT: ${fmtRut(node.rut)}</div>` : '<div class="detail-rut" style="color:#555570;">Sin RUT</div>'}
            <span class="detail-type ${node.type}">${node.type === 'person' ? '👤 Persona' : '🏢 Empresa'}</span>
        </div>
    `;

    if (node.names && node.names.length > 1) {
        html += `
            <div class="detail-block">
                <h3>🏷️ Variaciones del nombre</h3>
                <div style="font-size:12px;color:#c8c8d8;line-height:1.6;">
                    ${node.names.map(n => `<div>• ${esc(n)}</div>`).join('')}
                </div>
            </div>
        `;
    }

    // SII details for companies
    if (node.type === 'company' && node.rut) {
        try {
            const res = await fetch(`/api/company/${node.rut}`);
            const sii = await res.json();
            if (sii.found) {
                html += `
                    <div class="detail-block">
                        <h3>📊 Información SII</h3>
                        <div class="detail-row"><span class="detail-label">Tramo Ventas</span><span class="detail-value">${sii.tramo_ventas || '—'}</span></div>
                        <div class="detail-row"><span class="detail-label">Trabajadores</span><span class="detail-value">${sii.n_trabajadores ?? '—'}</span></div>
                        <div class="detail-row"><span class="detail-label">Actividad</span><span class="detail-value">${sii.actividad || '—'}</span></div>
                        <div class="detail-row"><span class="detail-label">Capital Propio</span><span class="detail-value">${sii.tramo_capital || '—'}</span></div>
                        <div class="detail-row"><span class="detail-label">Inicio</span><span class="detail-value">${sii.fecha_inicio || '—'}</span></div>
                    </div>
                `;
            }
        } catch (err) { console.error('SII error:', err); }
    }

    // Connections
    if (connections.length > 0) {
        html += `
            <div class="detail-block">
                <h3>🔗 Conexiones (${connections.length})</h3>
                ${connections.map(c => `
                    <div class="connection-item" onclick="expandNode(${c.node.id})">
                        <span class="connection-dot ${c.node.type}"></span>
                        <span class="connection-name">${esc(c.node.name)}</span>
                        ${c.count > 1 ? `<span class="connection-badge">×${c.count}</span>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Source documents
    if (documents.length > 0) {
        const seen = new Set();
        const uniqueDocs = documents.filter(d => {
            if (seen.has(d.url)) return false;
            seen.add(d.url); return true;
        }).slice(0, 10);

        html += `
            <div class="detail-block">
                <h3>📄 Documentos Fuente (${uniqueDocs.length})</h3>
                ${uniqueDocs.map(d => `
                    <a class="doc-link" href="${d.url}" target="_blank" title="${d.url}">
                        <span class="doc-date">${d.date || ''}</span>CVE ${d.cve}
                    </a>
                `).join('')}
            </div>
        `;
    }

    content.innerHTML = html;
}

// ── Stats ──────────────────────────────────────────────────────────────────
function updateStats() {
    const nNodes = Object.keys(allGraphData.nodes).length;
    const nEdges = Object.keys(allGraphData.edges).length;
    const people = Object.values(allGraphData.nodes).filter(n => n.type === 'person').length;
    const companies = Object.values(allGraphData.nodes).filter(n => n.type === 'company').length;
    const bar = document.getElementById('statsBar');
    document.getElementById('statsText').textContent =
        `${nNodes} nodos · ${nEdges} conexiones · ${people} personas · ${companies} empresas`;
    bar.classList.remove('hidden');
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}

function esc(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function capitalize(str) {
    if (!str) return '';
    return str.replace(/\b\w/g, c => c.toUpperCase());
}

function fmtRut(rut) {
    if (!rut) return '';
    const parts = rut.split('-');
    if (parts.length === 2) {
        const num = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, '.');
        return `${num}-${parts[1]}`;
    }
    return rut;
}

// ── Keyboard Shortcuts ──────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Both Delete and Backspace (mostly for Mac users)
    if (e.key === 'Delete' || e.key === 'Backspace') {
        // Do nothing if no network, or if user is typing in the search box
        if (!network || document.activeElement === searchInput) return;

        const selectedNodes = network.getSelectedNodes();
        const selectedEdges = network.getSelectedEdges();
        let modified = false;

        if (selectedNodes.length > 0) {
            selectedNodes.forEach(nodeId => {
                delete allGraphData.nodes[nodeId];

                // Find and remove all connected edges from both data structures
                const connectedEdges = network.getConnectedEdges(nodeId);
                connectedEdges.forEach(edgeId => {
                    const visEdge = edgesDataset.get(edgeId);
                    if (visEdge && visEdge.edgeKey) {
                        delete allGraphData.edges[visEdge.edgeKey];
                    }
                    edgesDataset.remove(edgeId);
                });

                // Finally remove the node itself
                nodesDataset.remove(nodeId);
            });
            modified = true;
        } else if (selectedEdges.length > 0) {
            // If they only selected an edge directly, allow deleting just that edge
            selectedEdges.forEach(edgeId => {
                const visEdge = edgesDataset.get(edgeId);
                if (visEdge && visEdge.edgeKey) {
                    delete allGraphData.edges[visEdge.edgeKey];
                    edgesDataset.remove(edgeId);
                    modified = true;
                }
            });
        }

        if (modified) {
            // Close the detail panel since what they were looking at might be gone
            const detailSection = document.getElementById('detailSection');
            const instructions = document.getElementById('instructions');
            if (detailSection) detailSection.classList.add('hidden');
            if (instructions) instructions.style.display = 'block';
            
            updateStats();
        }
    }
});