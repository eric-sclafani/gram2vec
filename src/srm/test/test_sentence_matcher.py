import spacy
import json
from dataclasses import dataclass
from srm import linearize_tree


@dataclass
class TestSentence:
    truth:str
    text:str

template = [
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("FALSE", ""),
    TestSentence("FALSE", ""),
    TestSentence("FALSE", ""),
]

it_cleft_sents = [
    TestSentence("TRUE", "It was Jane’s car that got stolen last night"),
    TestSentence("TRUE", "It wasn’t the most obvious problem that intrigued me, it was the subtle issue of responsibility."),
    TestSentence("TRUE", "It was a moral debt that I had inherited from my grandmother."),
    TestSentence("TRUE", "It was in the principate of Tiberius Caesar that their druids and prophets and healers of this type were abolished."),
    TestSentence("TRUE", "It was during the principate of Tiberius Caesar when their druids and prophets and healers of this type were abolished."),
    TestSentence("TRUE", "If it were John who is the candidate, I would vote for him."),
    TestSentence("FALSE", "He was a man who ate my cakes."),
    TestSentence("FALSE", "It was in the principate of Tiberius Caesar, who reigned from AD14 to AD37."),
    TestSentence("FALSE", "It wasn’t the most obvious problem."),
]

pseudo_cleft_sents = [
    TestSentence("TRUE", "What I want is some peace and quiet!"),
    TestSentence("TRUE", "What you need to do is to rest for a while."),
    TestSentence("TRUE", "Where I want to go is a place so far away from here."),
    TestSentence("TRUE", "How she paid for her food was with her credit card."),
    TestSentence("TRUE", "Some peace and quiet is what I want."),
    TestSentence("TRUE", "A place so far away from here is where I want to go."),
    TestSentence("FALSE", "I want a hamburger."),
    TestSentence("FALSE", "Where did I put that potato?"),
    TestSentence("FALSE", "I like having peace and quiet"),
    TestSentence("FALSE", "What I need is none of your business."),
]

all_cleft_sents = [
    TestSentence("TRUE", "All Jimmy wants for Christmas is a brand new bicycle."),
    TestSentence("TRUE", "All the girl does is complain about everything"),
    TestSentence("TRUE", "Was all she wanted a good job?"),
    TestSentence("TRUE", "Was all Jeff saw blue and yellow?"),
    TestSentence("TRUE", "Was all she saw blue and yellow?"),
    TestSentence("TRUE", "Was all the dog saw blue and yellow?"),
    TestSentence("TRUE", "Was all Jeff saw the moon?"),
    TestSentence("TRUE", "All the dog in the tree knew was that the bone was on the grass."),
    TestSentence("TRUE", "While riding her bike, all Sarah thought about was seeing her friends at the ball game."),
    TestSentence("FALSE", "I want all of these shirts"),
    TestSentence("FALSE", "The boxes were all filled with potatoes"),
    TestSentence("FALSE", "In all of this time, I have never seen a goose."),
    TestSentence("FALSE", "The man riding the bike wanted to play baseball"),
    TestSentence("FALSE", "In the morning, Jack and Ryan played with all of the toys.")
]

there_cleft_sents = [
    TestSentence("TRUE", "There's this orphan kid that I'm trying to adopt."),
    TestSentence("TRUE", "There is a new car Camille wanted to buy"),
    TestSentence("TRUE", "There are five ducks swimming in the pool"),
    TestSentence("TRUE", "There is a new James Bond movie that I want to watch tomorrow."),
    TestSentence("TRUE", "Is there a new TV show that Mary wants to watch?"),
    TestSentence("FALSE","I want to walk over there"),
    TestSentence("FALSE", "Cindy and Mindy go there everyday."),
    TestSentence("FALSE", "In the future, I will eat lunch over there"),
    TestSentence("FALSE", "There is a unicorn in the garden"),
]

if_because_cleft_sents = [
    TestSentence("TRUE", "If he wants to be a millionaire, it's because he wants to help poor children"),
    TestSentence("TRUE", "If it seems that she is meddling, it's just because she's just trying to help the family."),
    TestSentence("TRUE", "If the dog in the tree is crying, it's only because he is hungry."),
    TestSentence("TRUE", "If Sam and Tom are married, it's because they're in love"),
    TestSentence("TRUE", "If she wanted to drive to Florida, it's because she hates flying."),
    TestSentence("TRUE", "If I ate the chicken, it's because I like meat."),
    TestSentence("FALSE", "If I wanted to, I could go to the mall"),
    TestSentence("FALSE", "Because I am young, I can do anything"),
    TestSentence("FALSE", "Sarah ate the ham sandwhich because she was hungry"),
]

passive_sents = [
    TestSentence("TRUE", "The lion was killed by the hunter."),
    TestSentence("TRUE", "English is spoken all over the world"),
    TestSentence("TRUE", "The book which was given to me by Mary was really interesting."),
    TestSentence("TRUE", "The book given to me by Mary was really interesting"),
    TestSentence("TRUE", "The windows have been cleaned"),
    TestSentence("TRUE", "The work will be finished soon"),
    TestSentence("TRUE", "They might have been invited to the party"),
    TestSentence("TRUE", "John might have been invited to the party"),
    TestSentence("FALSE", "The cat is chasing the rat"),
    TestSentence("FALSE", "He is always doing things by the books"),
    TestSentence("FALSE", "The dwarf is sitting by the fire"),
]


sub_clauses_sents = [
    
    TestSentence("TRUE", "The chicken, who was busy laying eggs, sat happily."),
    TestSentence("TRUE", "Steven, who was looking over the hill, sighed wistfully."),
    TestSentence("TRUE", "The man in the house, who waited patiently, stepped outside."),
    TestSentence("TRUE", "Watching Star Wars, which has lots of special effects, is my favorite thing to do."),
    
    TestSentence("TRUE", "The girl sighed wistfully, looking over the hill."),
    TestSentence("TRUE", "The man sat in his house, waiting patiently for the rain to stop"),
    
    TestSentence("TRUE", "Singing her favorite song, Jasmine skipped along the road."),
    TestSentence("TRUE", "Looking over the hill, she sighed wistfully."),
    TestSentence("TRUE", "If I can find my wallet, we can all go for ice cream"),
    
    TestSentence("TRUE (PARSE ERROR, IGNORE)", "Sitting happily, the chicken laid eggs."),
    TestSentence("TRUE (SKIP FOR NOW)", "Jeff was mad at whoever egged his house."),
    #TestSentence("FALSE", ""),
    #TestSentence("FALSE", ""),
    #TestSentence("FALSE", ""),
]

subj_rel_clause_sents = [
    TestSentence("TRUE", "J. K. Rowling is the author who wrote the Harry Potter books"),
    TestSentence("TRUE", "The students who live next door make too much noise"),
    TestSentence("TRUE", "There are numerous viruses which cause the common cold"),
    TestSentence("TRUE", "There are numerous viruses that cause the common cold"),
    TestSentence("TRUE", "I bought a new car that is very fast"),
    TestSentence("TRUE", "I like the woman who lives next door"),
    TestSentence("TRUE", "I sent a letter which arrived three weeks later"),
    TestSentence("FALSE (OBJ REL)", "Last year, someone who I know had a book published"),
    TestSentence("FALSE (OBJ REL)", "She is the author that I have interviewed"),
    TestSentence("FALSE (OBJ REL)", "The beach that we visited last week is closed"),
    TestSentence("FALSE (OBJ REL)", "The man who I spoke to had a thick accent"),
    TestSentence("FALSE (OBJ REL)", "This is a problem which we know nothing about"),
    TestSentence("FALSE (OBJ REL)", "These are the documents that you need to enter the country"),
    TestSentence("FALSE", "He hates eating oranges and I love that about him"),
    TestSentence("FALSE", "Who is that man in the gray jacket?"),
    TestSentence("FALSE", "That Jessica rides her bike fast intrigues me"),
]

obj_rel_clause_sents = [
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("TRUE", ""),
    TestSentence("FALSE", ""),
    TestSentence("FALSE", ""),
    TestSentence("FALSE", ""),
]

nlp = spacy.load("en_core_web_lg")



def save_test_sentences(sentences, fname):
    with open(f"testing_output/{fname}.txt", "w") as fout:
        for test in sentences:
            doc = nlp(test.text)
            for sent in doc.sents:
                fout.write(f"{test.truth}\n{test.text}\n{linearize_tree(sent)}\n\n")
                








        
        
        
def main():
    pass

# BACKUPS
# https://regex101.com/r/TgWuWz/1

if __name__ == "__main__":
    main()