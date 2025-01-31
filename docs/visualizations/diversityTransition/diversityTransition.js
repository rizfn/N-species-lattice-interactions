async function loadData() {
    const timeseriesPath = '../../data/diversityTransition/timeseries/N_10-100_sparsity_0.9_gamma_0.1/timeseries.csv';
    const connectivityMatrixPath = '../../data/diversityTransition/timeseries/N_10-100_sparsity_0.9_gamma_0.1/connectivity_matrix.csv';
    const reactionRatesMatrixPath = '../../data/diversityTransition/timeseries/N_10-100_sparsity_0.9_gamma_0.1/reaction_rates_matrix.csv';

    const [timeseries, connectivityMatrixText, reactionRatesMatrixText] = await Promise.all([
        d3.csv(timeseriesPath),
        d3.text(connectivityMatrixPath),
        d3.text(reactionRatesMatrixPath)
    ]);

    const connectivityMatrix = d3.csvParseRows(connectivityMatrixText, d => d.map(Number));
    const reactionRatesMatrix = d3.csvParseRows(reactionRatesMatrixText, d => d.map(Number));

    timeseries.forEach(d => {
        Object.keys(d).forEach(key => {
            d[key] = +d[key];
        });
    });

    return { timeseries, connectivityMatrix, reactionRatesMatrix };
}

// Define an async function to handle the top-level await
(async () => {
    const { timeseries, connectivityMatrix, reactionRatesMatrix } = await loadData();
    console.log(timeseries);
    console.log(connectivityMatrix);
    console.log(reactionRatesMatrix);

    const timeseriesSvgMargin = { top: 20, right: 20, bottom: 30, left: 50 };

    const timeseriesSvg = d3.select('body').append('svg')
        .attr('width', "49vw") // Use viewport width
        .attr('height', "100vh") // Use viewport height
        .append('g')
        .attr('transform', `translate(${timeseriesSvgMargin.left}, ${timeseriesSvgMargin.top})`);

    const width = window.innerWidth * 0.5 - timeseriesSvgMargin.left - timeseriesSvgMargin.right;
    const height = window.innerHeight - timeseriesSvgMargin.top - timeseriesSvgMargin.bottom;

    const x = d3.scaleLinear()
        .domain(d3.extent(timeseries, d => d.time))
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(timeseries, d => d3.max(Object.values(d).slice(2)))])
        .range([height, 0]);

    const line = d3.line()
        .x(d => x(d.time))
        .y(d => y(d.value));

    timeseriesSvg.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(d3.axisBottom(x));

    timeseriesSvg.append('g')
        .call(d3.axisLeft(y));

    const chemicals = Object.keys(timeseries[0]).slice(2);

    const colorScale = d3.scaleSequential()
        .domain([0, chemicals.length - 1])
        .interpolator(d3.interpolateRainbow);

    chemicals.forEach((chemical, index) => {
        const chemicalData = timeseries.map(d => ({ time: d.time, value: d[chemical] }));
        timeseriesSvg.append('path')
            .datum(chemicalData)
            .attr('fill', 'none')
            .attr('stroke', colorScale(index))
            .attr('stroke-width', 1.5)
            .attr('d', line);
    });

    // Add the present-time line
    let presentTime = timeseries[999].time;
    const presentTimeLine = timeseriesSvg.append('line')
        .attr('x1', x(presentTime))
        .attr('x2', x(presentTime))
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', 'grey')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 2');

    // Add an invisible, wider stroke for easier dragging
    const dragArea = timeseriesSvg.append('line')
        .attr('x1', x(presentTime))
        .attr('x2', x(presentTime))
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', 'transparent')
        .attr('stroke-width', 10)
        .attr('cursor', 'ew-resize');

    // Drag behavior
    const drag = d3.drag()
        .on('drag', function (event) {
            const newX = Math.max(0, Math.min(width, event.x));
            presentTime = x.invert(newX);
            presentTimeLine
                .attr('x1', newX)
                .attr('x2', newX);
            dragArea
                .attr('x1', newX)
                .attr('x2', newX);
            drawNetworkDiagram(presentTime);
        });

    // Apply drag behavior to the present-time line
    dragArea.call(drag);

    // Click behavior
    d3.select('svg').on('click', function (event) {
        const [mouseX] = d3.pointer(event);
        const newX = Math.max(0, Math.min(width, mouseX - timeseriesSvgMargin.left));
        presentTime = x.invert(newX);
        presentTimeLine
            .attr('x1', newX)
            .attr('x2', newX);
        dragArea
            .attr('x1', newX)
            .attr('x2', newX);
        drawNetworkDiagram(presentTime);
    });

    // Determine the maximum value in the connectivity matrix for the color scale
    const N_resources = d3.max(connectivityMatrix.flat());

    const resourcesColorScale = d3.scaleSequential()
    .domain([0, N_resources])
    // .interpolator(d3.interpolateRainbow);
    .interpolator(d3.interpolateRgb("purple", "orange"));


    function drawNetworkDiagram(presentTime) {
        const N_resources = d3.max(connectivityMatrix.flat());

        const survivalThreshold = 0.01 / chemicals.length;
        const closestTime = timeseries.reduce((prev, curr) => Math.abs(curr.time - presentTime) < Math.abs(prev.time - presentTime) ? curr : prev);
        const currentData = timeseries.find(d => d.time === closestTime.time);
        let survivingSpecies = chemicals.filter(chemical => currentData[chemical] >= survivalThreshold);

        const nodes = survivingSpecies.map(chemical => ({
            id: chemical,
            size: currentData[chemical] * 200 // Scale node size
        }));

        survivingSpecies = survivingSpecies.map(chemical => parseInt(chemical.replace('chemical', '')));

        const links = [];
        survivingSpecies.forEach(source => {
            survivingSpecies.forEach(target => {
                if (connectivityMatrix[source][target] >= 0) {
                    links.push({
                        source: `chemical${source}`,
                        target: `chemical${target}`,
                        weight: 5 * Math.pow(reactionRatesMatrix[source][target], 2),
                        color: connectivityMatrix[source][target]
                    });
                }
            });
        });

        // Create a new SVG for the network diagram if it doesn't exist
        let networkSvg = d3.select('#network-diagram');
        if (networkSvg.empty()) {
            networkSvg = d3.select('body').append('svg')
                .attr('id', 'network-diagram')
                .attr('width', "49vw")
                .attr('height', "100vh")
                .append('g')
                .attr('transform', `translate(${timeseriesSvgMargin.left}, ${timeseriesSvgMargin.top})`);

            // Define arrow marker
            networkSvg.append('defs').append('marker')
                .attr('id', 'arrow')
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 10)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-4L10,0L0,4')
                .attr('fill', 'black');
        } else {
            // Ensure defs and marker are present
            if (networkSvg.select('defs').empty()) {
                networkSvg.append('defs').append('marker')
                    .attr('id', 'arrow')
                    .attr('viewBox', '0 -5 10 10')
                    .attr('refX', 10)
                    .attr('refY', 0)
                    .attr('markerWidth', 6)
                    .attr('markerHeight', 6)
                    .attr('orient', 'auto')
                    .append('path')
                    .attr('d', 'M0,-5L10,0L0,5')
                    .attr('fill', 'black');
            }
        }

        const radius = Math.min(width, height) / 2;
        const angleStep = (2 * Math.PI) / nodes.length;

        nodes.forEach((node, index) => {
            node.x = width / 2 + radius * Math.cos(index * angleStep);
            node.y = height / 2 + radius * Math.sin(index * angleStep);
        });

        // Update nodes
        const nodeSelection = networkSvg.selectAll('circle')
            .data(nodes, d => d.id);

        nodeSelection.enter().append('circle')
            .attr('r', 0)
            .attr('fill', d => colorScale(chemicals.indexOf(d.id)))
            .attr('stroke', 'black')
            .attr('stroke-width', 1.5)
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .transition()
            .duration(1000)
            .attr('r', d => d.size);

        nodeSelection.transition()
            .duration(1000)
            .attr('r', d => d.size)
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        nodeSelection.exit().transition()
            .duration(1000)
            .attr('r', 0)
            .remove();

        // Update links
        const linkSelection = networkSvg.selectAll('line')
            .data(links, d => `${d.source}-${d.target}`);

        linkSelection.enter().append('line')
            .attr('marker-end', 'url(#arrow)')
            .attr('x1', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
                return sourceNode.x + sourceNode.size * Math.cos(angle);
            })
            .attr('y1', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
                return sourceNode.y + sourceNode.size * Math.sin(angle);
            })
            .attr('x2', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(sourceNode.y - targetNode.y, sourceNode.x - targetNode.x);
                return sourceNode.x + sourceNode.size * Math.cos(angle);
            })
            .attr('y2', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(sourceNode.y - targetNode.y, sourceNode.x - targetNode.x);
                return sourceNode.y + sourceNode.size * Math.sin(angle);
            })
            .transition()
            .duration(1000)
            .attr('stroke-width', d => d.weight)
            .attr('stroke', d => resourcesColorScale(d.color))
            .attr('x2', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(sourceNode.y - targetNode.y, sourceNode.x - targetNode.x);
                return targetNode.x + targetNode.size * Math.cos(angle);
            })
            .attr('y2', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(sourceNode.y - targetNode.y, sourceNode.x - targetNode.x);
                return targetNode.y + targetNode.size * Math.sin(angle);
            });

        linkSelection.transition()
            .duration(1000)
            .attr('stroke-width', d => d.weight)
            .attr('x1', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
                return sourceNode.x + sourceNode.size * Math.cos(angle);
            })
            .attr('y1', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(targetNode.y - sourceNode.y, targetNode.x - sourceNode.x);
                return sourceNode.y + sourceNode.size * Math.sin(angle);
            })
            .attr('x2', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(sourceNode.y - targetNode.y, sourceNode.x - targetNode.x);
                return targetNode.x + targetNode.size * Math.cos(angle);
            })
            .attr('y2', d => {
                const sourceNode = nodes.find(n => n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target);
                const angle = Math.atan2(sourceNode.y - targetNode.y, sourceNode.x - targetNode.x);
                return targetNode.y + targetNode.size * Math.sin(angle);
            });

        linkSelection.exit().transition()
            .duration(1000)
            .attr('stroke-width', 0)
            .remove();
    }

    // Initial draw of the network diagram
    drawNetworkDiagram(presentTime);

})();