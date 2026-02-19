# Recommendation Engine

Collaborative filtering recommendation system using user-item matrix factorization.

## Features
- User-based collaborative filtering
- Item-based collaborative filtering  
- Cosine similarity scoring
- Top-N recommendations

## Usage
```python
from recommender import RecommendationEngine
engine = RecommendationEngine()
engine.add_rating("user1", "item1", 5.0)
recs = engine.recommend("user1", n=5)
```

## Algorithms
Supports both user-based and item-based collaborative filtering with cosine similarity.
