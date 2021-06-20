# python3

# refer to https://www.youtube.com/watch?v=OViaWp9Q-Oc

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
    vertex_count, edge_count = map(int, input().split())
    graph = FlowGraph(vertex_count)
    for _ in range(edge_count):
        u, v, capacity = map(int, input().split())
        graph.add_edge(u - 1, v - 1, capacity)
    return graph


if __name__ == '__main__':
    graph = read_data()
    print(graph.solve(0, graph.size() - 1))


# 5 7
# 1 2 2
# 2 5 5
# 1 3 6
# 3 4 2
# 4 5 1
# 3 2 3
# 2 4 1