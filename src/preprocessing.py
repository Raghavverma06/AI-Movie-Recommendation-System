import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

class DataIngestor:
    """
    Handles data ingestion and generation of realistic synthetic movie dataset
    for 100% offline capabilities.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.movies_path = os.path.join(self.data_dir, "movies.csv")
        self.ratings_path = os.path.join(self.data_dir, "ratings.csv")

    def load_or_create_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the movies and ratings datasets. Generates them synthetically if missing.
        """
        if os.path.exists(self.movies_path) and os.path.exists(self.ratings_path):
            try:
                movies_df = pd.read_csv(self.movies_path)
                ratings_df = pd.read_csv(self.ratings_path)
                return movies_df, ratings_df
            except Exception as e:
                print(f"Error loading files: {e}. Re-generating dataset.")
        
        return self._generate_synthetic_dataset()

    def _generate_synthetic_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates a highly realistic synthetic movie and rating dataset.
        Contains 1000 movies with detailed metadata and 100,000 ratings from 600 users.
        """
        print("Generating realistic synthetic movie dataset offline...")
        
        # 1. Define seed directors, cast, keywords, and templates per genre
        genres_list = ["Action", "Adventure", "Sci-Fi", "Drama", "Romance", 
                       "Comedy", "Thriller", "Horror", "Mystery", "Fantasy"]
        
        directors = {
            "Action": ["Christopher Nolan", "James Cameron", "George Miller", "Michael Bay", "John Woo"],
            "Adventure": ["Steven Spielberg", "Peter Jackson", "Ridley Scott", "George Lucas"],
            "Sci-Fi": ["Denis Villeneuve", "Christopher Nolan", "Lana Wachowski", "Ridley Scott", "James Cameron"],
            "Drama": ["Martin Scorsese", "Frank Darabont", "David Fincher", "Quentin Tarantino", "Francis Ford Coppola"],
            "Romance": ["Richard Linklater", "James Cameron", "Marc Webb", "Rob Reiner"],
            "Comedy": ["Edgar Wright", "Todd Phillips", "Adam McKay", "Judd Apatow"],
            "Thriller": ["David Fincher", "Alfred Hitchcock", "Christopher Nolan", "Denis Villeneuve", "Martin Scorsese"],
            "Horror": ["James Wan", "Jordan Peele", "Ari Aster", "John Carpenter", "Wes Craven"],
            "Mystery": ["David Fincher", "Rian Johnson", "Christopher Nolan", "Denis Villeneuve"],
            "Fantasy": ["Peter Jackson", "David Yates", "Chris Columbus", "Guillermo del Toro"]
        }
        
        cast_pool = {
            "Action": ["Leonardo DiCaprio", "Christian Bale", "Tom Cruise", "Keanu Reeves", "Scarlett Johansson", "Robert Downey Jr.", "Hugh Jackman", "Charlize Theron", "Chris Hemsworth"],
            "Adventure": ["Harrison Ford", "Elijah Wood", "Ian McKellen", "Sam Neill", "Tom Hanks", "Johnny Depp", "Emma Watson", "Daniel Radcliffe", "Orlando Bloom"],
            "Sci-Fi": ["Matthew McConaughey", "Sigourney Weaver", "Ryan Gosling", "Keanu Reeves", "Sandra Bullock", "Matt Damon", "Zoe Saldana", "Mark Hamill", "Carrie Fisher"],
            "Drama": ["Robert De Niro", "Al Pacino", "Morgan Freeman", "Brad Pitt", "Meryl Streep", "Kate Winslet", "Tim Robbins", "Matthew McConaughey", "Joaquin Phoenix"],
            "Romance": ["Kate Winslet", "Leonardo DiCaprio", "Ryan Gosling", "Rachel McAdams", "Ethan Hawke", "Julie Delpy", "Hugh Grant", "Julia Roberts"],
            "Comedy": ["Jim Carrey", "Will Ferrell", "Steve Carell", "Jonah Hill", "Ryan Reynolds", "Seth Rogen", "Emma Stone", "Ben Stiller", "Paul Rudd"],
            "Thriller": ["Brad Pitt", "Morgan Freeman", "Edward Norton", "Jodie Foster", "Christian Bale", "Leonardo DiCaprio", "Jake Gyllenhaal", "Ben Affleck"],
            "Horror": ["Toni Collette", "Lupita Nyong'o", "Jamie Lee Curtis", "Florence Pugh", "Ethan Hawke", "Patrick Wilson", "Vera Farmiga"],
            "Mystery": ["Daniel Craig", "Hugh Jackman", "Jake Gyllenhaal", "Amy Adams", "Leonardo DiCaprio", "Christian Bale", "Robert Downey Jr."],
            "Fantasy": ["Elijah Wood", "Daniel Radcliffe", "Emma Watson", "Johnny Depp", "Ian McKellen", "Orlando Bloom", "Viggo Mortensen"]
        }
        
        keywords_pool = {
            "Action": ["superhero", "explosion", "heist", "fight", "car chase", "martial arts", "revenge", "rescue"],
            "Adventure": ["treasure hunt", "journey", "island", "wilderness", "quest", "exploration", "survival", "ancient"],
            "Sci-Fi": ["space travel", "artificial intelligence", "dystopia", "time travel", "alien", "cyborg", "future", "simulation"],
            "Drama": ["family struggle", "mental illness", "friendship", "tragedy", "ambition", "historical events", "betrayal", "hope"],
            "Romance": ["love story", "star-crossed lovers", "forbidden love", "second chance", "secret crush", "destiny", "relationship"],
            "Comedy": ["slapstick", "satire", "parody", "misunderstanding", "buddy road trip", "coming of age", "high school"],
            "Thriller": ["serial killer", "conspiracy", "hostage", "psychological", "kidnapping", "corruption", "ticking clock"],
            "Horror": ["haunted house", "possession", "monster", "survival horror", "zombie", "ghost", "slasher", "curse"],
            "Mystery": ["detective", "puzzle", "murder mystery", "secret identity", "twist ending", "disappearance", "conspiracy"],
            "Fantasy": ["magic", "wizards", "dragons", "prophecy", "mythology", "alternate realm", "elves", "dark lord"]
        }

        # Real seed movies to ground the data in reality (approx 60)
        seed_movies = [
            # Sci-Fi / Action
            {"title": "Inception", "genres": "Action|Sci-Fi|Thriller", "director": "Christopher Nolan", "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page", "keywords": "dream, subconscious, heist, simulation, architecture", "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
            {"title": "The Dark Knight", "genres": "Action|Crime|Drama", "director": "Christopher Nolan", "cast": "Christian Bale, Heath Ledger, Aaron Eckhart", "keywords": "superhero, joker, justice, chaos, corruption", "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
            {"title": "Interstellar", "genres": "Adventure|Drama|Sci-Fi", "director": "Christopher Nolan", "cast": "Matthew McConaughey, Anne Hathaway, Jessica Chastain", "keywords": "space travel, black hole, wormhole, survival, time dilation", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival on a dying Earth."},
            {"title": "The Matrix", "genres": "Action|Sci-Fi", "director": "Lana Wachowski", "cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss", "keywords": "simulation, artificial intelligence, cyberpunk, prophecy, hacker", "description": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence."},
            {"title": "Avatar", "genres": "Action|Adventure|Fantasy|Sci-Fi", "director": "James Cameron", "cast": "Sam Worthington, Zoe Saldana, Sigourney Weaver", "keywords": "alien, moon, colonization, military, nature", "description": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."},
            {"title": "Titanic", "genres": "Drama|Romance", "director": "James Cameron", "cast": "Leonardo DiCaprio, Kate Winslet, Billy Zane", "keywords": "shipwreck, love story, upper class, tragedy, historical events", "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."},
            {"title": "Gladiator", "genres": "Action|Adventure|Drama", "director": "Ridley Scott", "cast": "Russell Crowe, Joaquin Phoenix, Connie Nielsen", "keywords": "rome, gladiator, revenge, emperor, arena", "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."},
            {"title": "Blade Runner 2049", "genres": "Action|Drama|Sci-Fi", "director": "Denis Villeneuve", "cast": "Ryan Gosling, Harrison Ford, Ana de Armas", "keywords": "replicant, cyberpunk, artificial intelligence, detective, memory", "description": "A new blade runner, LAPD Officer K, unearths a long-buried secret that has the potential to plunge what's left of society into chaos."},
            {"title": "Dune", "genres": "Action|Adventure|Drama|Sci-Fi", "director": "Denis Villeneuve", "cast": "Timothee Chalamet, Rebecca Ferguson, Zendaya", "keywords": "desert, spice, prophecy, empire, space opera", "description": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir is troubled by visions of a dark future."},
            {"title": "Pulp Fiction", "genres": "Crime|Drama", "director": "Quentin Tarantino", "cast": "John Travolta, Uma Thurman, Samuel L. Jackson", "keywords": "hitman, non-linear, boxing, drugs, crime anthology", "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
            {"title": "The Lord of the Rings: The Fellowship of the Ring", "genres": "Adventure|Fantasy", "director": "Peter Jackson", "cast": "Elijah Wood, Ian McKellen, Orlando Bloom", "keywords": "magic ring, wizards, elves, quest, friendship", "description": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron."},
            {"title": "Jurassic Park", "genres": "Action|Adventure|Sci-Fi", "director": "Steven Spielberg", "cast": "Sam Neill, Laura Dern, Jeff Goldblum", "keywords": "dinosaurs, theme park, genetic engineering, escape, survival", "description": "A pragmatic paleontologist touring an almost complete theme park on an island in Central America is tasked with protecting a couple of kids after a power failure causes the park's cloned dinosaurs to run loose."},
            {"title": "The Wolf of Wall Street", "genres": "Biography|Comedy|Crime|Drama", "director": "Martin Scorsese", "cast": "Leonardo DiCaprio, Jonah Hill, Margot Robbie", "keywords": "finance, stock market, corruption, drugs, wall street", "description": "Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker living the high life to his fall involving crime, corruption and the federal government."},
            {"title": "Shutter Island", "genres": "Mystery|Thriller", "director": "Martin Scorsese", "cast": "Leonardo DiCaprio, Mark Ruffalo, Ben Kingsley", "keywords": "mental asylum, detective, twist ending, conspiracy, trauma", "description": "Teddy Daniels and Chuck Aule, two US marshals, are sent to an asylum on a remote island in order to investigate the disappearance of a patient, where Teddy uncovers a shocking truth."},
            {"title": "Seven", "genres": "Crime|Mystery|Thriller", "director": "David Fincher", "cast": "Brad Pitt, Morgan Freeman, Gwyneth Paltrow", "keywords": "serial killer, seven deadly sins, detective, rainy city, twist ending", "description": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives."},
            {"title": "Fight Club", "genres": "Drama", "director": "David Fincher", "cast": "Brad Pitt, Edward Norton, Helena Bonham Carter", "keywords": "insomnia, alter ego, anti-consumerism, secret society, violence", "description": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more."},
            {"title": "The Shawshank Redemption", "genres": "Drama", "director": "Frank Darabont", "cast": "Tim Robbins, Morgan Freeman, Bob Gunton", "keywords": "prison, prison break, hope, friendship, corruption", "description": "Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion."},
            {"title": "Forrest Gump", "genres": "Drama|Romance", "director": "Robert Zemeckis", "cast": "Tom Hanks, Robin Wright, Gary Sinise", "keywords": "historical events, simpleton, love, running, friendship", "description": "The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man with an IQ of 75, who yearns to be reunited with his childhood sweetheart."},
            {"title": "Toy Story", "genres": "Adventure|Comedy|Fantasy", "director": "John Lasseter", "cast": "Tom Hanks, Tim Allen, Don Rickles", "keywords": "toys, friendship, rivalry, childhood, animation", "description": "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom."},
            {"title": "The Avengers", "genres": "Action|Sci-Fi", "director": "Joss Whedon", "cast": "Robert Downey Jr., Chris Evans, Scarlett Johansson", "keywords": "superhero, team-up, alien invasion, shield, comic book", "description": "Earth's mightiest heroes must come together and learn to fight as a team if they are to stop the mischievous Loki and his alien army from enslaving humanity."},
            {"title": "Django Unchained", "genres": "Drama|Western", "director": "Quentin Tarantino", "cast": "Jamie Foxx, Christoph Waltz, Leonardo DiCaprio", "keywords": "slavery, bounty hunter, revenge, shootout, old west", "description": "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner."},
            {"title": "The Shining", "genres": "Drama|Horror", "director": "Stanley Kubrick", "cast": "Jack Nicholson, Shelley Duvall, Danny Lloyd", "keywords": "haunted hotel, isolation, writer, madness, winter", "description": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings from both past and future."},
            {"title": "The Conjuring", "genres": "Horror|Mystery|Thriller", "director": "James Wan", "cast": "Patrick Wilson, Vera Farmiga, Ron Livingston", "keywords": "paranormal investigator, haunted house, demon, exorcism, true story", "description": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse."},
            {"title": "Get Out", "genres": "Horror|Mystery|Thriller", "director": "Jordan Peele", "cast": "Daniel Kaluuya, Allison Williams, Bradley Whitford", "keywords": "social satire, conspiracy, hypnosis, racism, escape", "description": "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness about their reception eventually reaches a boiling point."},
            {"title": "A Beautiful Mind", "genres": "Biography|Drama", "director": "Ron Howard", "cast": "Russell Crowe, Ed Harris, Jennifer Connelly", "keywords": "schizophrenia, mathematician, codebreaking, love, Nobel prize", "description": "After John Nash, a brilliant but asocial mathematician, accepts secret work in cryptography, his life takes a turn for the nightmarish."},
            {"title": "The Lion King", "genres": "Adventure|Drama|Fantasy", "director": "Roger Allers", "cast": "Matthew Broderick, Jeremy Irons, James Earl Jones", "keywords": "savannah, lion, uncle betrayal, coming of age, musical", "description": "Lion prince Simba and his father are targeted by his bitter uncle Scar, who wants to ascend the throne himself."},
            {"title": "Good Will Hunting", "genres": "Drama|Romance", "director": "Gus Van Sant", "cast": "Matt Damon, Robin Williams, Ben Affleck", "keywords": "genius, janitor, therapy, math, Boston, friendship", "description": "Will Hunting, a janitor at M.I.T., has a gift for mathematics, but needs help from a psychologist in order to find direction in his life."},
            {"title": "The Silence of the Lambs", "genres": "Crime|Drama|Thriller", "director": "Jonathan Demme", "cast": "Jodie Foster, Anthony Hopkins, Lawrence A. Bonney", "keywords": "fbi agent, serial killer, cannibalism, prison interview, profiling", "description": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims."},
            {"title": "Joker", "genres": "Crime|Drama|Thriller", "director": "Todd Phillips", "cast": "Joaquin Phoenix, Robert De Niro, Zazie Beetz", "keywords": "origin story, mental illness, social isolation, clown, rebellion", "description": "In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime."},
            {"title": "The Truman Show", "genres": "Comedy|Drama", "director": "Peter Weir", "cast": "Jim Carrey, Laura Linney, Ed Harris", "keywords": "reality tv, simulated town, paranoia, hidden cameras, escape", "description": "An insurance salesman discovers his whole life is actually a reality TV show watched by millions of viewers around the globe."},
        ]

        # Populate year
        for i, m in enumerate(seed_movies):
            m["movieId"] = i + 1
            m["release_year"] = random.randint(1990, 2023)
            # Add dynamic fields if missing
            if "keywords" not in m:
                m["keywords"] = "classic, cinema, masterwork"
            if "description" not in m:
                m["description"] = "A highly rated film with stunning performances and a gripping storyline."
        
        movies_data = seed_movies.copy()
        current_id = len(movies_data) + 1
        
        # 2. Expand database procedurally to 1000 movies
        # Verb lists for description generation
        verbs = ["travels to", "uncovers", "battles", "falls in love with", "confronts", "discovers", "escapes from", "investigates", "navigates"]
        nouns = ["a mysterious artifact", "a deadly conspiracy", "a forbidden romance", "a digital virtual world", "a post-apocalyptic wasteland", "their deepest fears", "a hidden empire", "a secret agent"]
        adjectives = ["dangerous", "haunting", "spectacular", "gripping", "breathtaking", "unforgettable", "mysterious", "futuristic"]
        
        # Let's generate 970 more movies
        movie_titles_template = [
            ("The Last {}", "Adventure"),
            ("Shadow of the {}", "Thriller"),
            ("Return of the {}", "Fantasy"),
            ("Love in {}", "Romance"),
            ("The {} Project", "Sci-Fi"),
            ("Chronicles of {}", "Adventure"),
            ("Secrets of the {}", "Mystery"),
            ("The {} Escape", "Action"),
            ("Diary of a {}", "Drama"),
            ("{} Night", "Horror"),
            ("How to {}", "Comedy"),
            ("The {} Syndicate", "Crime"),
            ("Beyond the {}", "Sci-Fi"),
            ("Curse of the {}", "Horror"),
            ("Tales of {}", "Fantasy"),
            ("The {} Equation", "Sci-Fi"),
            ("Hearts in {}", "Romance"),
            ("The {} Heist", "Action"),
            ("Whispers of the {}", "Thriller"),
            ("Chasing {}", "Comedy")
        ]
        
        filler_words = ["Galaxies", "Empire", "Love", "Paris", "AI", "Cyborg", "Pharaoh", "High School", "Deep Sea", "Haunting", "Algorithm", "Shadows", "Wilderness", "Symphony", "Destiny", "Dreamer", "Revenge", "Artifact", "Dimension", "Time"]
        
        while len(movies_data) < 1000:
            template, base_genre = random.choice(movie_titles_template)
            filler = random.choice(filler_words)
            title = template.format(filler)
            
            # Avoid exact duplicate titles
            if any(m["title"] == title for m in movies_data):
                title = f"{title} II"
                
            # Assign random set of 1-3 genres
            g_count = random.randint(1, 3)
            selected_genres = [base_genre]
            while len(selected_genres) < g_count:
                other_g = random.choice(genres_list)
                if other_g not in selected_genres:
                    selected_genres.append(other_g)
            
            genres_str = "|".join(selected_genres)
            primary_genre = base_genre
            
            # Pick director
            dir_list = directors.get(primary_genre, ["Independent Director"])
            director = random.choice(dir_list)
            
            # Pick cast
            cast_list = cast_pool.get(primary_genre, ["Actor Alpha", "Actor Beta"])
            c_sample = random.sample(cast_list, min(3, len(cast_list)))
            cast_str = ", ".join(c_sample)
            
            # Pick keywords
            kw_list = keywords_pool.get(primary_genre, ["movie", "story"])
            kw_sample = random.sample(kw_list, min(4, len(kw_list)))
            kw_str = ", ".join(kw_sample)
            
            # Generate description
            verb = random.choice(verbs)
            noun = random.choice(nouns)
            adj = random.choice(adjectives)
            description = f"In this {adj} tale, a main protagonist {verb} {noun} that threatens to change everything they know."
            
            release_year = random.randint(1975, 2024)
            
            movies_data.append({
                "movieId": current_id,
                "title": title,
                "genres": genres_str,
                "director": director,
                "cast": cast_str,
                "keywords": kw_str,
                "description": description,
                "release_year": release_year
            })
            current_id += 1
            
        movies_df = pd.DataFrame(movies_data)
        movies_df.to_csv(self.movies_path, index=False)
        print(f"Generated {len(movies_df)} movies successfully.")

        # 3. Generate ratings
        # 600 users, 100,000 ratings.
        # To make collaborative filtering output realistic, let's create user sub-cohorts.
        # Cohorts have specific genre preferences.
        print("Generating 100,000 collaborative ratings from 600 users...")
        
        ratings_data = []
        user_profiles = []
        
        for u in range(1, 601):
            # Assign favorite genres to this user profile
            fav_genres = random.sample(genres_list, k=random.randint(1, 3))
            disliked_genres = random.sample([g for g in genres_list if g not in fav_genres], k=random.randint(1, 2))
            
            # General rating generosity of this user
            user_bias = random.uniform(-0.6, 0.6)
            
            user_profiles.append({
                "userId": u,
                "fav": fav_genres,
                "dislike": disliked_genres,
                "bias": user_bias
            })
            
        # Calculate rating probability and scores
        movie_avg_qualities = {row["movieId"]: random.uniform(2.5, 4.5) for _, row in movies_df.iterrows()}
        movie_id_to_genres = {row["movieId"]: row["genres"].split("|") for _, row in movies_df.iterrows()}
        
        # We will generate ~100,000 ratings.
        # Average ratings per user is 1000 movies * ~16% = 166 ratings
        # Let's generate rating entries iteratively
        all_movie_ids = movies_df["movieId"].tolist()
        
        ratings_records = []
        
        # Ensure every movie gets at least 8 ratings so NearestNeighbors runs fine
        for movie_id in all_movie_ids:
            # pick random users to rate it
            voters = random.sample(range(1, 601), k=random.randint(8, 20))
            for u in voters:
                profile = user_profiles[u-1]
                movie_genres = movie_id_to_genres[movie_id]
                
                # Base score from movie average quality
                score = movie_avg_qualities[movie_id]
                
                # Boost if user's favorite genre matches
                for fg in profile["fav"]:
                    if fg in movie_genres:
                        score += 0.6
                
                # Penalize if user dislikes this genre
                for dg in profile["dislike"]:
                    if dg in movie_genres:
                        score -= 0.8
                        
                # Add user's specific rating bias and small noise
                score += profile["bias"] + random.normalvariate(0, 0.4)
                
                # Clip rating to [1.0, 5.0] and round to half-star or whole-star
                rating = np.clip(score, 1.0, 5.0)
                rating = round(rating * 2) / 2.0  # round to nearest 0.5
                
                ratings_records.append({
                    "userId": u,
                    "movieId": movie_id,
                    "rating": rating,
                    "timestamp": random.randint(900000000, 1600000000)
                })
                
        # Fill in the rest randomly to reach exactly 100,000 ratings
        current_ratings_count = len(ratings_records)
        target_ratings = 100000
        
        # Keep track of existing user-movie pairs to avoid duplicate ratings
        existing_pairs = set((r["userId"], r["movieId"]) for r in ratings_records)
        
        print(f"Base ratings populated: {current_ratings_count}. Adding remaining to reach {target_ratings}...")
        
        attempts = 0
        max_attempts = 1000000
        while len(ratings_records) < target_ratings and attempts < max_attempts:
            attempts += 1
            u = random.randint(1, 600)
            m = random.choice(all_movie_ids)
            
            if (u, m) not in existing_pairs:
                profile = user_profiles[u-1]
                movie_genres = movie_id_to_genres[m]
                
                score = movie_avg_qualities[m]
                for fg in profile["fav"]:
                    if fg in movie_genres:
                        score += 0.6
                for dg in profile["dislike"]:
                    if dg in movie_genres:
                        score -= 0.8
                score += profile["bias"] + random.normalvariate(0, 0.4)
                
                rating = np.clip(score, 1.0, 5.0)
                rating = round(rating * 2) / 2.0
                
                ratings_records.append({
                    "userId": u,
                    "movieId": m,
                    "rating": rating,
                    "timestamp": random.randint(900000000, 1600000000)
                })
                existing_pairs.add((u, m))

        ratings_df = pd.DataFrame(ratings_records)
        # Sort values for a neat look
        ratings_df = ratings_df.sort_values(by=["userId", "movieId"]).reset_index(drop=True)
        ratings_df.to_csv(self.ratings_path, index=False)
        
        print(f"Generated {len(ratings_df)} ratings successfully.")
        return movies_df, ratings_df


class DataPreprocessor:
    """
    Cleans text features for Content-Based Filtering and
    pivots ratings for Collaborative Filtering.
    """
    @staticmethod
    def preprocess_metadata(movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines genres, description, keywords, cast, and director
        into a cleaned 'metadata_soup' column.
        """
        df = movies_df.copy()
        
        # Fill missing values with empty string
        df["genres"] = df["genres"].fillna("").apply(lambda x: x.replace("|", " "))
        df["director"] = df["director"].fillna("")
        df["cast"] = df["cast"].fillna("")
        df["keywords"] = df["keywords"].fillna("")
        df["description"] = df["description"].fillna("")
        
        # Normalize text attributes: clean whitespace, lowercase directors/cast to prevent split errors
        def clean_word(text: str) -> str:
            return text.lower().replace(" ", "")

        def clean_list(text: str) -> str:
            # Splits comma separated items, strips whitespace, converts to lowercase and joins without space
            return " ".join([clean_word(item) for item in text.split(",")])

        df["cleaned_director"] = df["director"].apply(clean_word)
        df["cleaned_cast"] = df["cast"].apply(clean_list)
        df["cleaned_keywords"] = df["keywords"].apply(clean_list)
        df["cleaned_genres"] = df["genres"].apply(lambda x: x.lower())
        
        # Combine everything into a metadata soup
        df["metadata_soup"] = (
            df["cleaned_genres"] + " " +
            df["cleaned_director"] + " " +
            df["cleaned_cast"] + " " +
            df["cleaned_keywords"] + " " +
            df["description"].apply(lambda x: x.lower())
        )
        
        # Strip duplicate spaces
        df["metadata_soup"] = df["metadata_soup"].apply(lambda x: " ".join(x.split()))
        return df

    @staticmethod
    def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates user-item interaction pivot table.
        Rows are userIds, columns are movieIds. Missing values are filled with 0.
        """
        return ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    @staticmethod
    def split_data(ratings_df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits ratings into train and test sets for collaborative filtering evaluation.
        We do this by randomly sampling fraction of records for validation.
        """
        shuffled = ratings_df.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled) * (1 - test_size))
        train_df = shuffled.iloc[:split_idx]
        test_df = shuffled.iloc[split_idx:]
        return train_df, test_df
