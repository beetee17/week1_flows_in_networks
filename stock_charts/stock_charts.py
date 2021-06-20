# python3

class Edge:

    def __init__(self, from_, to, capacity):
        self.from_ = from_
        self.to = to
        self.capacity = capacity
        self.flow = 0
        self.residual = None
    
    def isResidual(self):
        return self.capacity == 0

    def getRemainingCapacity(self):
        return self.capacity - self.flow
    
    def augment(self, bottleneck):
        self.flow += bottleneck
        self.residual.flow -= bottleneck
    
class FlowGraph:

    def __init__(self, n):
        
        self.n = n
        
        self.graph = [[] for _ in range(n)]

        # visited flag for each node in the graph for 
        # use in breadth-first search
        # a node, i, is visited if the corresponding value at
        # visited[i] == visitedToken
        # mark nodes as unvisited in O(1) time by incrementing
        # the value of visitedToken
        self.visitedToken = 1
        self.visited =[0 for i in range(n)]

        self.maxFlow = 0

    def add_edge(self, from_, to, capacity):
      
        forward_edge = Edge(from_, to, capacity)
        backward_edge = Edge(to, from_, 0)

        forward_edge.residual = backward_edge
        backward_edge.residual = forward_edge

        self.graph[from_].append(forward_edge)
        self.graph[to].append(backward_edge)

    def size(self):
        return len(self.graph)

    def markNodesUnvisited(self):
        self.visitedToken += 1
    
    def visit(self, node):
        self.visited[node] = self.visitedToken
    
    def was_visited(self, node):
        return self.visited[node] == self.visitedToken
    
    def solve(self, source, sink):

        start = True
        flow = 0

        while start or flow != 0:
            
            start = False
            self.markNodesUnvisited()
            flow = self.bfs(source, sink)
            self.maxFlow += flow
            
        return self.maxFlow
    
    def bfs(self, source, sink):
        from queue import Queue

        q = Queue(maxsize=self.n)
        self.visit(source)
        q.put_nowait(source)

        prev = [None for i in range(self.n)]
        while not q.empty():

            node = q.get_nowait()
            # print('checking node', node)

            if node == sink: break

            # print('node not sink')
       
            for edge in self.graph[node]:
                # print('check edge from', edge.from_, 'to', edge.to)

                cap = edge.getRemainingCapacity()
                # print('capacity', edge.capacity)
                # print('flow', edge.flow)
                # print('remaining', cap)

                if cap > 0 and not self.was_visited(edge.to):
                    self.visit(edge.to)
                    # print('visiting node', edge.to)
                    prev[edge.to] = edge
                    q.put_nowait(edge.to)
                #     print('visiting neighbour', edge.to)
                
                # elif self.was_visited(edge.to):
                #     print('visited node', edge.to, 'already')
        
        if not prev[sink]: return 0 # sink was not reachable

        bottleneck = 10e9
        edge = prev[sink]

        while edge:
            # print('retracing from', edge.to, 'to', edge.from_)
            # print('bottleneck', bottleneck)
            # print('remaining', edge.getRemainingCapacity())
            bottleneck = min(bottleneck, edge.getRemainingCapacity())
            edge = prev[edge.from_]
        
        edge = prev[sink]

        while edge:
            edge.augment(bottleneck)
            edge = prev[edge.from_]

        return bottleneck


def read_data():
    num_stocks, num_points = map(int, input().split())
    stock_data = [list(map(int, input().split())) for i in range(num_stocks)]
    return stock_data, num_stocks, num_points

def get_graph(stock_data, num_stocks, num_points):
    # Create a bipartite graph with 2 nodes for each stock
    # If stock(i) < stock(j) (all the k points of stock(i) is less 
    # than all the k points of stock(j)), add an edge between the nodes
    # Convert it to a max flow problem by adding source and sink and capacities of 1 unit.

    graph = FlowGraph(2*num_stocks + 2)
    source = 0
    sink = graph.size() - 1

    for v in range(1, num_stocks+1):
        graph.add_edge(source, v, 1)
    
    for u in range(num_stocks+1, 2*num_stocks+1):
        graph.add_edge(u, sink, 1)

    for i in range(len(stock_data)):
        for j in range(len(stock_data)):
            fits = True
            for k in range(num_points):
                if stock_data[i][k] <= stock_data[j][k]:
                    fits = False
                    break
            
            if fits:
                graph.add_edge(i+1, num_stocks+j+1, 1)
    

    return graph, num_stocks


def get_matches(graph, num_stocks):

    source = 0
    sink = graph.size() - 1
    matches = [-1 for i in range(num_stocks)]
    num_matches = 0

    for i in range(graph.size()):
        for edge in graph.graph[i]:

            if edge.flow == 1 and edge.from_ != source and edge.to != sink:
                
                matches[edge.from_-1] = edge.to - num_stocks
                num_matches += 1

    return matches, num_matches

if __name__ == '__main__':
    stock_data, num_stocks, num_points = read_data()
    graph, num_stocks = get_graph(stock_data, num_stocks, num_points)
    graph.solve(0, graph.size() - 1)
    matches, num_matches = get_matches(graph, num_stocks)
    # After computing max flow, the min number of charts required is the 
    # no. of sockes - no. of matches
    print(num_stocks-num_matches)


