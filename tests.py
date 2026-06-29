from recommender import RecommendationEngine

def test_add_and_recommend():
    e = RecommendationEngine()
    e.add_ratings_bulk([("a","m1",5),("a","m2",3),("b","m1",4),("b","m3",5)])
    recs = e.recommend("a")
    assert any(item == "m3" for item, _ in recs)

def test_similarity():
    e = RecommendationEngine()
    sim = e._cosine_similarity({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert abs(sim - 1.0) < 0.01

if __name__ == "__main__":
    test_add_and_recommend()
    test_similarity()
    print("All tests passed!")
