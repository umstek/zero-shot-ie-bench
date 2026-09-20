"""Question pools for the mixed benchmarks (source for bench_spectrum.py).

Classification: sentiment (24 texts) and topic (12 texts) pools; NER pool
(18 texts) with gold span sets. Questions are authored across subjects and
signal strength — from one strong signal up to sarcasm, negation flips,
lowercase brands, and context-dependent ambiguity (Apple the company vs
apple the fruit). Gold labels stay unambiguous to a careful human.

The pool dicts are keyed by authoring group only, so the design notes stay
traceable; bench_spectrum.py flattens them into ONE mixed pool per ability
and measures each question's difficulty after all systems have answered.
"""

from __future__ import annotations

TIERS = ("easy", "medium", "hard")

SENTIMENT: dict[str, list[tuple[str, str]]] = {
    "easy": [
        ("I absolutely love this playlist; every song is perfect.", "positive"),
        ("The hotel exceeded all our expectations.", "positive"),
        ("Worst customer service I have ever experienced.", "negative"),
        ("The delivery arrived broken and two weeks late.", "negative"),
        ("The shop opens at nine and closes at six.", "neutral"),
        ("The lecture covered three chapters of the textbook.", "neutral"),
        ("What a wonderful surprise, thank you so much!", "positive"),
        ("The room was dirty and the air conditioning was broken.", "negative"),
    ],
    "medium": [
        ("The camera is great, but the battery, screen, and support all disappoint.", "negative"),
        ("Service was slow and the place was noisy, though the dessert was good.", "negative"),
        ("The course was demanding, but I learned a lot and enjoyed it.", "positive"),
        ("Not the cheapest option, but easily the most reliable laptop I've owned.", "positive"),
        ("The film runs for three hours and won two awards.", "neutral"),
        ("The update is 400 MB and installs in about five minutes.", "neutral"),
        ("My order arrived on time, but two items were missing.", "negative"),
        ("The hotel is clean, quiet, and close to the station.", "positive"),
    ],
    "hard": [
        ("Oh great, another meeting that could have been an email.", "negative"),
        ("Wow, the app crashed exactly when I needed it. Perfect timing.", "negative"),
        ("I don't hate the new design; it's just worse in every way.", "negative"),
        ("Sure, ignore my emails for two weeks, that's fine, I love being ignored.", "negative"),
        ("Well, that presentation wasn't a total disaster - only most of it.", "negative"),
        ("I'm thrilled to pay a restocking fee for a product that never worked.", "negative"),
        ("It's not a bad book at all, honestly one of the better ones this year.", "positive"),
        ("The instructions were so clear that I assembled it wrong twice.", "negative"),
    ],
}

TOPIC: dict[str, list[tuple[str, str]]] = {
    "easy": [
        ("The new GPU architecture doubles throughput per watt.", "technology"),
        ("The striker scored twice in the final minutes of the match.", "sports"),
        ("The senate passed the budget bill after a late-night session.", "politics"),
        ("The company reported record quarterly revenue driven by overseas sales.", "business"),
        ("Developers reported memory leaks in the latest framework release.", "technology"),
        ("She won the marathon with a personal best of 2:19.", "sports"),
        ("Voters head to the polls in the regional elections next month.", "politics"),
        ("Shares fell three percent after the merger announcement.", "business"),
    ],
    "medium": [
        ("The football club's shares rose five percent after the transfer deal.", "business"),
        ("The midfielder's agent negotiated a record signing bonus for the striker.", "sports"),
        ("Lawmakers questioned the social media CEO about the data breach.", "politics"),
        ("The new graphics card sells out within minutes of every restock.", "technology"),
        ("The central bank raised interest rates before the housing report.", "business"),
        ("The startup's AI pitch won the national innovation award.", "technology"),
        ("The striker donated his match bonus to a local hospital.", "sports"),
        ("The senate debated new rules for autonomous vehicle testing.", "politics"),
    ],
    "hard": [
        ("The government launched a national AI strategy to boost chip manufacturing.", "politics"),
        ("The esports team's IPO was oversubscribed on its first trading day.", "business"),
        ("Biotech stocks fell after the regulator rejected the new therapy.", "business"),
        ("The Olympic committee partnered with a crypto exchange for the games.", "sports"),
        ("The prime minister's speech went viral on the new video app.", "politics"),
        ("The gaming console outsold every streaming device this holiday season.", "technology"),
        ("The chess world championship was streamed live to millions online.", "sports"),
        ("Venture funding for climate-tech startups doubled after the summit.", "business"),
    ],
}

NER: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {
    "easy": [
        ("Serena Williams won the match in Melbourne on Saturday.",
         [("Serena Williams", "person"), ("Melbourne", "location")]),
        ("Amazon opened a new office in Berlin last month.",
         [("Amazon", "company"), ("Berlin", "location")]),
        ("The iPhone 15 Pro costs 999 dollars in the United States.",
         [("iPhone 15 Pro", "product"), ("United States", "location")]),
        ("Elon Musk visited the factory in Texas with engineers from Tesla.",
         [("Elon Musk", "person"), ("Texas", "location"), ("Tesla", "company")]),
        ("Toyota recalled the Corolla in Canada after reports of brake faults.",
         [("Toyota", "company"), ("Corolla", "product"), ("Canada", "location")]),
        ("Angela Merkel spoke at the conference in Munich.",
         [("Angela Merkel", "person"), ("Munich", "location")]),
    ],
    "medium": [
        ("He transferred the money via revolut to pay for the spotify subscription.",
         [("revolut", "company"), ("spotify", "company")]),
        ("The galaxy note series competes directly with the iphone in most markets.",
         [("galaxy note series", "product"), ("iphone", "product")]),
        ("musk's spacex launched another batch of starlink satellites from florida.",
         [("musk", "person"), ("spacex", "company"), ("starlink", "product"),
          ("florida", "location")]),
        ("She ordered a big mac and a coke at the mcdonald's near times square.",
         [("big mac", "product"), ("coke", "product"), ("mcdonald's", "company"),
          ("times square", "location")]),
        ("The windows 11 update broke outlook for thousands of excel users at ford.",
         [("windows 11", "product"), ("outlook", "product"), ("excel", "product"),
          ("ford", "company")]),
        ("Zhang yiming founded bytedance in beijing before creating tiktok.",
         [("Zhang yiming", "person"), ("bytedance", "company"),
          ("beijing", "location"), ("tiktok", "product")]),
    ],
    "hard": [
        ("Tim Cook said Apple would open stores in India while eating an "
         "apple at a farm near Delhi.",
         [("Tim Cook", "person"), ("Apple", "company"), ("India", "location"),
          ("Delhi", "location")]),
        ("While visiting Amazon in Seattle, Jeff Bezos talked about Alexa, "
         "Kindle, and the rainforest tour he took near Manaus.",
         [("Amazon", "company"), ("Seattle", "location"), ("Jeff Bezos", "person"),
          ("Alexa", "product"), ("Kindle", "product"), ("Manaus", "location")]),
        ("The London office of Goldman Sachs hired Marie Curie's "
         "granddaughter, a physicist from Paris.",
         [("London", "location"), ("Goldman Sachs", "company"),
          ("Marie Curie", "person"), ("Paris", "location")]),
        ("Microsoft's Satya Nadella and Google's Sundar Pichai met at the "
         "White House to discuss AI.",
         [("Microsoft", "company"), ("Satya Nadella", "person"),
          ("Google", "company"), ("Sundar Pichai", "person"),
          ("White House", "location")]),
        ("After leaving DeepMind, the researcher joined OpenAI's London "
         "team to work on ChatGPT.",
         [("DeepMind", "company"), ("OpenAI", "company"),
          ("London", "location"), ("ChatGPT", "product")]),
        ("Nike signed a deal with fc barcelona to make jerseys for the "
         "2026 season.",
         [("Nike", "company"), ("fc barcelona", "company")]),
    ],
}


def validate_ner_pools() -> None:
    """Every gold span must occur exactly once in its text."""
    for tier in TIERS:
        for text, truth in NER[tier]:
            for span, _ in truth:
                if text.count(span) != 1:
                    raise ValueError(f"gold span not unique in text: "
                                     f"{span!r} ({tier})")


if __name__ == "__main__":
    validate_ner_pools()
    n_cls = sum(len(v) for v in SENTIMENT.values()) + sum(
        len(v) for v in TOPIC.values())
    print(f"pools OK: {n_cls} classification + "
          f"{sum(len(v) for v in NER.values())} NER questions")
