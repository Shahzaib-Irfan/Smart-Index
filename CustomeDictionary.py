class CustomDictionary:
    def __init__(self, initial_capacity=1024):
        self.capacity = initial_capacity
        self.size = 0
        # Initialize buckets with None
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75

    def _hash(self, key):
        """Custom hash function for string and number keys."""
        if isinstance(key, str):
            # For strings, use polynomial rolling hash
            hash_value = 0
            p = 31  # prime number for better distribution
            m = self.capacity
            for char in key:
                hash_value = (hash_value * p + ord(char)) % m
            return hash_value
        else:
            # For other types, use their hash value
            return hash(key) % self.capacity

    def _resize(self):
        """Resize the hash table when load factor exceeds threshold."""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        
        # Rehash all existing key-value pairs
        for bucket in old_buckets:
            for key, value in bucket:
                self[key] = value

    def __setitem__(self, key, value):
        """Set key-value pair in the dictionary."""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        # Check if key exists and update value
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return
        
        # If key doesn't exist, add new key-value pair
        bucket.append((key, value))
        self.size += 1
        
        # Check if resize is needed
        if self.size / self.capacity >= self.load_factor_threshold:
            self._resize()

    def __getitem__(self, key):
        """Get value for given key."""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        raise KeyError(key)

    def __delitem__(self, key):
        """Delete key-value pair from dictionary."""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self.size -= 1
                return
        raise KeyError(key)

    def get(self, key, default=None):
        """Get value for key, return default if key doesn't exist."""
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        """Clear all items from dictionary."""
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

    def items(self):
        """Return list of all (key, value) pairs."""
        result = []
        for bucket in self.buckets:
            result.extend(bucket)
        return result

    def keys(self):
        """Return list of all keys."""
        return [key for key, _ in self.items()]

    def values(self):
        """Return list of all values."""
        return [value for _, value in self.items()]

    def __len__(self):
        """Return number of items in dictionary."""
        return self.size

    def __contains__(self, key):
        """Check if key exists in dictionary."""
        try:
            self[key]
            return True
        except KeyError:
            return False

class CustomDefaultDict(CustomDictionary):
    def __init__(self, default_factory=None, initial_capacity=1024):
        super().__init__(initial_capacity)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if self.default_factory is None:
                raise KeyError(key)
            self[key] = self.default_factory()
            return self[key]