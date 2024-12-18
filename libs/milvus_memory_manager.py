import hashlib
import json
from pymilvus import MilvusClient
import time

class MilvusMemoryManager:
    def __init__(self, db_path="./data/memory.db", collection_name="memory_collection"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.memory_client = self.init_milvus()

    def init_milvus(self):
        memory_client = MilvusClient(self.db_path)
        if not memory_client.has_collection(self.collection_name):
            memory_client.create_collection(collection_name=self.collection_name, dimension=384, auto_id=True)
        return memory_client

    def dump_to_milvus(self, embeddings, memory, source):
        # hash memory
        memory_hash = hashlib.sha256(json.dumps(memory).encode()).hexdigest()
        # hash as int64
        id = int(memory_hash, 16)
        timestamp = memory['timestamp']
        
        # TODO: change source to speaker or something
        data_to_insert = [
            {
                "vector": embedding,
                "memory_hash": memory_hash,
                "source": source,
                "timestamp": timestamp
            } for embedding in embeddings
        ]
        
        res = self.memory_client.insert(collection_name=self.collection_name, data=data_to_insert)
        return res
    
    def get_memories(self, embeddings, top_k=10, minimum_age=10):
        
        # min age is in minutes
        minimum_timstamp = time.time() - minimum_age * 60
        res = self.memory_client.search(
            collection_name=self.collection_name,
            data=embeddings,
            output_fields=["memory_hash", "source", "timestamp"],
            filter={"range": {"timestamp": {"lt": minimum_timstamp}}},
            limit=top_k
        )
        result = []

        # get a list of unique hashes from res and the smallest distance for each hash
        unique_hashes = []

        distances = {}
        for memory in res[0]:

            memory_hash = memory["entity"]["memory_hash"]
            distance = memory["distance"]

            if memory_hash not in unique_hashes:
                unique_hashes.append(memory_hash)
                distances[memory_hash] = distance
            else:
                if distance < distances[memory_hash]:
                    distances[memory_hash] = distance

        for memory_hash in unique_hashes:
            with open(f"./data/memories/{memory_hash}.json", "r") as f:
                memory_json = json.load(f)
                #result.append(memory_json)
                result.append({
                    "summary": memory_json['summary_obj'],
                    "distance": distances[memory_hash]
                })
        return result