import heapq

# Create an empty heap
heap = []

# Add (push) elements into the heap
heapq.heappush(heap, 10)
heapq.heappush(heap, 1)
heapq.heappush(heap, 5)
heapq.heappush(heap, 7)

# Display the heap
print("Heap elements:", heap)

# Remove (pop) the smallest element
smallest = heapq.heappop(heap)
print("Smallest element popped:", smallest)

# Display the heap after popping
print("Heap after popping:", heap)
