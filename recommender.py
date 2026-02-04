import math
from collections import defaultdict

class RecommendationEngine:
    def __init__(self):
        self.ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)

    def add_rating(self, user, item, rating):
        self.ratings[user][item] = rating
        self.item_ratings[item][user] = rating

    def add_ratings_bulk(self, data):
        for user, item, rating in data:
            self.add_rating(user, item, rating)

    def _cosine_similarity(self, vec1, vec2):
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        dot = sum(vec1[k] * vec2[k] for k in common)
        mag1 = math.sqrt(sum(v**2 for k, v in vec1.items() if k in common))
        mag2 = math.sqrt(sum(v**2 for k, v in vec2.items() if k in common))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def _find_similar_users(self, target_user, n=10):
        similarities = []
        target_ratings = self.ratings.get(target_user, {})
        for user, ratings in self.ratings.items():
            if user != target_user:
                sim = self._cosine_similarity(target_ratings, ratings)
                if sim > 0:
                    similarities.append((user, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def recommend(self, target_user, n=5, method="user"):
        if method == "user":
            return self._user_based_recommend(target_user, n)
        return self._item_based_recommend(target_user, n)

    def _user_based_recommend(self, target_user, n):
        similar_users = self._find_similar_users(target_user)
        target_items = set(self.ratings.get(target_user, {}).keys())
        scores = defaultdict(float)
        weights = defaultdict(float)
        for user, sim in similar_users:
            for item, rating in self.ratings[user].items():
                if item not in target_items:
                    scores[item] += sim * rating
                    weights[item] += sim
        predictions = []
        for item in scores:
            if weights[item] > 0:
                predictions.append((item, scores[item] / weights[item]))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def _item_based_recommend(self, target_user, n):
        target_items = self.ratings.get(target_user, {})
        scores = defaultdict(float)
        for rated_item, rating in target_items.items():
            for other_item in self.item_ratings:
                if other_item not in target_items:
                    sim = self._cosine_similarity(self.item_ratings[rated_item], self.item_ratings[other_item])
                    scores[other_item] += sim * rating
        predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return predictions[:n]

if __name__ == "__main__":
    engine = RecommendationEngine()
    data = [("alice","movie1",5),("alice","movie2",3),("alice","movie3",4),("bob","movie1",4),("bob","movie2",2),("bob","movie4",5),("charlie","movie1",3),("charlie","movie3",5),("charlie","movie4",4)]
    engine.add_ratings_bulk(data)
    print("Recommendations for alice:", engine.recommend("alice"))
