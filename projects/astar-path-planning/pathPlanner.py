def do_a_star(grid, start, end, display_message):
    # Define the grid search space using the grid dimensions
    COL = len(grid)
    ROW = len(grid[0])

    # Ensure generated nodes lie within the grid boundaries
    def in_bounds(node):
        c, r = node
        return 0 <= c < COL and 0 <= r < ROW

    # Detect whether a cell is free or occupied, using 1 and 0 to indicate that
    def is_free(node):
        c, r = node
        return grid[c][r] == 1

    # Define a Heuristic function h(n) using euclidean distance
    def heuristic(node):
        return (((node[0] - end[0]) ** 2) + ((node[1] - end[1]) ** 2)) ** 0.5

    # Define all sets of moves to reach neighbours of a node
    moves = [(1,0),(-1,0),(0,1),(0,-1)]

    # Neighbours function which gathers all the neighbours of a given node
    def neighbours(node):
        c,r = node
        for dx,dy in moves:
            next = (c+dx,r+dy)
            if in_bounds(next) and is_free(next):
                yield next

    # Maintain dictionaries of g (real cost from start) and f (g + heuristic) for each node, initialise start with g = 0.0 and f = heuristic(start)
    g_score = {start:0.0}
    f_score = {start:heuristic(start)}

    # Store parent nodes to reconstruct path
    parents = {start:None}

    # OPEN contains nodes which are waiting to be explored
    OPEN = {start}

    # CLOSED contains nodes that have already been explored
    CLOSED = set()

    # Select a node from OPEN with the minimum cost f(n)
    def best_from_open(nodes):
        best = None
        best_f = None

        for node in nodes:
            f = f_score[node]

            if best is None or f < best_f:
                best = node
                best_f = f

            elif f == best_f and node == end:
                best = node

        return best


    # Begin the node search until the length of OPEN becomes empty, since there will be no more nodes to analyse
    while len(OPEN) > 0:

        # Remove the node with the smallest f(n) from OPEN for expansion
        current = best_from_open(OPEN)
        OPEN.remove(current)

        # If the selected node is the goal, reconstruct the optimal path using the parent dictionary
        if current == end:
            path=[current]

            while current != start:
                current = parents[current]
                path.append(current)

            path.reverse()
            return path

        # Mark the node as analysed by placing it in CLOSED
        CLOSED.add(current)

        # Look at all the neighbours of the node 
        for neighbour in neighbours(current):

            # The new g_score for each newly discovered node will be 1 more than the g_score for the parent node, since cost in all directions is +1 
            tentative_g = g_score[current] + 1

            # If the neighouring node is either new or has a lower g_score compared to the previously calculated one
            # update the dictionaries and record the path
            if neighbour not in g_score or tentative_g < g_score[neighbour]:

                parents[neighbour] = current
                g_score[neighbour] = tentative_g
                f_score[neighbour] = tentative_g + heuristic(neighbour)

                # If a better path is found to a previously expanded node, reopen it and add it to the queue
                if neighbour in CLOSED:
                    CLOSED.remove(neighbour)

                # Add newly discovered nodes to OPEN so they can be considered for expansion
                OPEN.add(neighbour)

    # If OPEN becomes empty before reaching the goal, no path exists
    display_message("No path found")

    return []