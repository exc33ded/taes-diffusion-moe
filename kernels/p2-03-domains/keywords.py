"""Domain keyword lists for TAES Step 2.3. Matched with word-boundary regex \\bkw\\b (case-insens.)
against the FULL synset label line (all comma-separated synonyms incl. Latin binomials).
A class is in a domain iff it matches >=1 keyword and 0 exclusion phrases. Both lists are written
into domains.json so the selection can never become unrecoverable again (results-log 2026-08-22)."""

VEHICLE_KW = [
    "cab", "taxi", "jeep", "limousine", "minivan", "van", "pickup", "truck", "trailer truck",
    "tow truck", "bus", "trolleybus", "minibus", "school bus", "streetcar", "tram", "locomotive",
    "train", "freight car", "passenger car", "bullet train", "steam locomotive", "electric locomotive",
    "motor scooter", "moped", "bicycle", "tricycle", "unicycle", "mountain bike", "bicycle-built-for-two",
    "motorcycle", "go-kart", "golfcart", "snowmobile", "snowplow", "forklift", "garbage truck",
    "moving van", "police van", "ambulance", "fire engine", "fire truck", "recreational vehicle",
    "camper", "airliner", "airship", "warplane", "military plane", "space shuttle", "helicopter",
    "balloon", "glider", "wing", "parachute", "canoe", "kayak", "gondola", "catamaran", "trimaran",
    "schooner", "yawl", "sailboat", "speedboat", "fireboat", "lifeboat", "liner", "container ship",
    "aircraft carrier", "submarine", "destroyer", "pirate ship", "paddle wheel", "riverboat", "steamboat",
    "cart", "shopping cart", "horse cart", "jinrikisha", "rickshaw", "sled", "bobsled", "dogsled",
    "wheelchair", "half track", "army tank", "amphibian", "convertible", "sports car", "race car",
    "beach wagon", "station wagon", "Model T", "oxcart",
]

ANIMAL_KW = [
    "fish", "shark", "ray", "stingray", "cock", "hen", "ostrich", "finch", "goldfinch", "brambling",
    "bird", "jay", "magpie", "chickadee", "ouzel", "kite", "eagle", "vulture", "owl", "salamander",
    "axolotl", "newt", "eft", "frog", "bullfrog", "toad", "turtle", "terrapin", "tortoise", "lizard",
    "gecko", "iguana", "chameleon", "whiptail", "agama", "Gila monster", "dragon", "alligator",
    "crocodile", "triceratops", "snake", "boa", "python", "cobra", "mamba", "viper", "rattlesnake",
    "sidewinder", "trilobite", "harvestman", "scorpion", "spider", "black widow", "tarantula", "tick",
    "centipede", "grouse", "ptarmigan", "prairie chicken", "peacock", "quail", "partridge", "parrot",
    "African grey", "macaw", "cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "merganser", "goose", "swan", "tusker", "echidna", "platypus",
    "wallaby", "koala", "wombat", "jellyfish", "anemone", "coral", "flatworm", "nematode", "conch",
    "snail", "slug", "chiton", "nautilus", "crab", "lobster", "crayfish", "isopod", "stork",
    "spoonbill", "flamingo", "heron", "egret", "bittern", "limpkin", "gallinule", "coot", "bustard",
    "turnstone", "sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican", "penguin",
    "albatross", "whale", "dolphin", "sea lion", "seal", "dog", "terrier", "hound", "spaniel",
    "retriever", "setter", "pointer", "shepherd", "sheepdog", "collie", "husky", "malamute", "samoyed",
    "pinscher", "schnauzer", "poodle", "chihuahua", "pug", "bulldog", "mastiff", "boxer", "corgi",
    "beagle", "foxhound", "coonhound", "greyhound", "whippet", "borzoi", "saluki", "wolfhound",
    "deerhound", "dachshund", "basset", "papillon", "spitz", "chow", "keeshond", "pomeranian",
    "griffon", "otterhound", "weimaraner", "vizsla", "ridgeback", "rottweiler", "doberman", "malinois",
    "groenendael", "kelpie", "komondor", "kuvasz", "leonberg", "newfoundland", "pyrenees",
    "saint bernard", "great dane", "bernese", "appenzeller", "entlebucher", "bouvier", "affenpinscher",
    "basenji", "lhasa", "shih-tzu", "maltese", "ibizan", "norwegian elkhound", "cairn", "sealyham",
    "airedale", "lakeland", "dandie", "scottish", "westie", "Pekinese", "bloodhound", "bluetick",
    "redbone", "schipperke", "briard", "Mexican hairless", "wolf", "coyote", "jackal", "dingo",
    "dhole", "fox", "hyena", "cat", "tabby", "lynx", "cougar", "leopard", "jaguar", "lion", "tiger",
    "cheetah", "bear", "mongoose", "meerkat", "beetle", "weevil", "ladybug", "fly", "bee", "ant",
    "grasshopper", "cricket", "walking stick", "cockroach", "mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
    "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber", "hare", "rabbit",
    "porcupine", "hedgehog", "squirrel", "marmot", "beaver", "guinea pig", "hamster", "sorrel",
    "zebra", "hog", "boar", "warthog", "ox", "buffalo", "bison", "ram", "bighorn", "ibex",
    "hartebeest", "impala", "gazelle", "camel", "llama", "weasel", "mink", "polecat", "ferret",
    "otter", "skunk", "badger", "armadillo", "sloth", "orangutan", "gorilla", "chimpanzee", "gibbon",
    "siamang", "guenon", "patas", "baboon", "macaque", "langur", "colobus", "monkey", "marmoset",
    "capuchin", "howler", "titi", "indri", "elephant", "panda", "barracouta", "eel", "coho",
    "rock beauty", "anemone fish", "sturgeon", "gar", "lionfish", "puffer", "horse", "dugong", "pig",
    "tench", "goldfish", "bulbul", "junco", "bunting", "robin", "thrush", "warbler", "starling",
    "vireo", "mockingbird", "dipper",
]

# Per-domain exclusions: classes whose labels contain a keyword but are not in the domain.
# "crane" is ambiguous (bird class 134 AND machine class 517) and is dropped from both domains.
ANIMAL_EXCLUDE = [
    "bee house", "dog sled", "dogsled", "feather boa", "horse cart", "cathode-ray", "spider web",
    "teddy", "snake fence", "hot dog", "coral reef", "horse chestnut", "coral fungus",
    "hen-of-the-woods", "crane",
]
VEHICLE_EXCLUDE = ["crane", "garden cart", "car mirror", "car wheel", "tank suit", "boat paddle"]
