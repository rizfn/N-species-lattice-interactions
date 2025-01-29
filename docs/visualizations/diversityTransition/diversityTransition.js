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
    let presentTime = timeseries[800].time;
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


    // Function to draw the network diagram
    function drawNetworkDiagram(presentTime) {
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
                        weight: 2 * Math.pow(reactionRatesMatrix[source][target], 2),
                        color: connectivityMatrix[source][target]
                    });
                }
            });
        });

        // Clear previous network diagram
        d3.select('#network-diagram').remove();

        // Create a new SVG for the network diagram
        const networkSvg = d3.select('body').append('svg')
            .attr('id', 'network-diagram')
            .attr('width', "49vw")
            .attr('height', "100vh")
            .append('g')
            .attr('transform', `translate(${timeseriesSvgMargin.left}, ${timeseriesSvgMargin.top})`);

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));

        const link = networkSvg.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke-width', d => d.weight)
            .attr('stroke', d => colorScale(d.color));

        const node = networkSvg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('r', d => d.size)
            .attr('fill', d => colorScale(chemicals.indexOf(d.id)))
            .attr('stroke', 'black')
            .attr('stroke-width', 1.5)
            .attr('cursor', 'grab')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('title')
            .text(d => d.id);

        simulation
            .nodes(nodes)
            .on('tick', ticked);

        simulation.force('link')
            .links(links);

        function ticked() {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }

    // Initial draw of the network diagram
    drawNetworkDiagram(presentTime);



})();