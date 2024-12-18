import random
from typing import Dict, Optional

class TypoGenerator:
    def __init__(self, 
                 error_rate: float = 0.15,
                 swap_rate: float = 0.5,    # How often to swap vs other errors
                 adjacent_bias: float = 0.7,  # How often to prefer adjacent keys vs random swaps
                 space_error_rate: float = 0.1,  # How often to introduce space errors
                 drop_rate: float = 0.2,    # How often to drop letters/words vs other errors
                 word_drop_rate: float = 0.1  # How often to drop entire words (rare)
                 ):
        """
        Initialize the typo generator with configurable error rates.
        
        Args:
            error_rate: Float between 0 and 1, probability of error per word
            swap_rate: Float between 0 and 1, probability of swap vs other errors
            adjacent_bias: Float between 0 and 1, probability of adjacent key vs random swap
            space_error_rate: Float between 0 and 1, probability of space errors
            drop_rate: Float between 0 and 1, probability of dropping letters vs other errors
            word_drop_rate: Float between 0 and 1, probability of dropping entire words
        """
        self.error_rate = error_rate
        self.swap_rate = swap_rate
        self.adjacent_bias = adjacent_bias
        self.space_error_rate = space_error_rate
        self.drop_rate = drop_rate
        self.word_drop_rate = word_drop_rate
        
        # Keyboard layout for adjacent-key errors
        self.keyboard_layout = {
            'q': ['w', 'a'],
            'w': ['q', 'e', 's'],
            'e': ['w', 'r', 'd'],
            'r': ['e', 't', 'f'],
            't': ['r', 'y', 'g'],
            'y': ['t', 'u', 'h'],
            'u': ['y', 'i', 'j'],
            'i': ['u', 'o', 'k'],
            'o': ['i', 'p', 'l'],
            'p': ['o', '['],
            'a': ['q', 's', 'z'],
            's': ['w', 'a', 'd', 'x'],
            'd': ['e', 's', 'f', 'c'],
            'f': ['r', 'd', 'g', 'v'],
            'g': ['t', 'f', 'h', 'b'],
            'h': ['y', 'g', 'j', 'n'],
            'j': ['u', 'h', 'k', 'm'],
            'k': ['i', 'j', 'l'],
            'l': ['o', 'k', ';'],
            'z': ['a', 'x'],
            'x': ['s', 'z', 'c'],
            'c': ['d', 'x', 'v'],
            'v': ['f', 'c', 'b'],
            'b': ['g', 'v', 'n'],
            'n': ['h', 'b', 'm'],
            'm': ['j', 'n'],
        }

        # Common letter swaps that occur due to typing patterns
        self.common_swaps = {
            'th': 'ht',    # the -> hte
            'ch': 'hc',    # which -> whihc
            'ph': 'hp',    # phone -> pohne
            'sh': 'hs',    # should -> shuold
            'wh': 'hw',    # what -> waht
            'ck': 'kc',    # back -> bakc
            're': 'er',    # were -> were
            'es': 'se',    # cases -> cases
            'on': 'no',    # one -> noe
            'in': 'ni',    # inside -> inisde
        }

    def _swap_adjacent_letter(self, word: str) -> str:
        """Swap two adjacent letters in a word."""
        if len(word) < 2:
            return word
            
        # Try up to 3 positions to find valid swap
        for _ in range(3):
            pos = random.randint(0, len(word) - 2)
            if word[pos] in self.keyboard_layout or word[pos + 1] in self.keyboard_layout:
                chars = list(word)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                return ''.join(chars)
        return word

    def _swap_with_keyboard_adjacent(self, word: str) -> str:
        """Replace a letter with an adjacent key."""
        if not word:
            return word
            
        chars = list(word)
        # Try up to 3 positions to find valid swap
        for _ in range(3):
            pos = random.randint(0, len(chars) - 1)
            char = chars[pos].lower()
            if char in self.keyboard_layout:
                new_char = random.choice(self.keyboard_layout[char])
                chars[pos] = new_char if chars[pos].islower() else new_char.upper()
                return ''.join(chars)
        return word

    def _apply_common_swap(self, word: str) -> str:
        """Apply common letter pattern swaps."""
        for pattern, swap in self.common_swaps.items():
            if pattern in word.lower():
                return word.replace(pattern, swap, 1)
        return word

    def _drop_letter(self, word: str) -> str:
        """Drop a random letter from the word."""
        if len(word) <= 1:
            return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]

    def _modify_spaces(self, text: str) -> str:
        """Add or remove spaces randomly."""
        if random.random() < 0.5:  # Double space
            parts = text.split(' ')
            if len(parts) < 2:
                return text
            pos = random.randint(0, len(parts) - 2)
            parts.insert(pos + 1, '')
            return ' '.join(parts)
        else:  # Remove space
            if ' ' not in text:
                return text
            return text.replace(' ', '', 1)

    def generate_typos(self, text: str) -> str:
        """
        Generate typos in the input text based on configured error rates.
        
        Args:
            text: Input string to modify
        
        Returns:
            String with introduced typing errors
        """
        words = text.split()
        result_words = []
        
        # First pass: handle word dropping
        words = [w for w in words if not (random.random() < self.word_drop_rate)]
        if not words:  # Ensure we don't drop all words
            return text
        
        # Second pass: handle other errors
        for word in words:
            # Only apply at most one error per word
            if random.random() < self.error_rate:
                if random.random() < self.drop_rate:
                    word = self._drop_letter(word)
                elif random.random() < self.swap_rate:  # Apply swap-based error
                    if random.random() < self.adjacent_bias:
                        # Try keyboard-adjacent swap first, fall back to adjacent letter swap
                        new_word = self._swap_with_keyboard_adjacent(word)
                        if new_word == word:  # If no keyboard swap worked
                            word = self._swap_adjacent_letter(word)
                        else:
                            word = new_word
                    else:
                        # Apply common pattern swap
                        word = self._apply_common_swap(word)
                else:  # Apply traditional typo
                    if random.random() < 0.5:
                        word = self._swap_adjacent_letter(word)
                    else:
                        word = self._swap_with_keyboard_adjacent(word)
            
            result_words.append(word)
        
        result_text = ' '.join(result_words)
        
        # Final pass: handle space errors
        if random.random() < self.space_error_rate:
            result_text = self._modify_spaces(result_text)
            
        return result_text

def main():    
    example_texts = [
    # Common workplace emails
    "I've attached the quarterly report for your review.",
    "Please let me know if you have any questions about the presentation.",
    "Could we schedule a meeting to discuss the project timeline?",
    "Thank you for your feedback on the proposal.",
    "I'll forward the information to the development team.",
    "Looking forward to our discussion tomorrow afternoon.",
    "The deadline for submission has been extended to Friday.",
    "Can you send me the updated version of the document?",
    
    # Technical writing
    "The function returns a boolean value indicating success.",
    "Make sure to initialize the variables before using them.",
    "The database migration should be completed overnight.",
    "Remember to commit your changes before pushing to main.",
    "Check the configuration settings in the environment file.",
    "The API documentation needs to be updated accordingly.",
    "Users should authenticate before accessing protected routes.",
    "The algorithm complexity is logarithmic in the worst case.",
    
    # Casual messages
    "Hey, are you still coming to lunch today?",
    "Did you see the new episode last night?",
    "I'm running a bit late, should be there in 15 minutes.",
    "Don't forget to bring your laptop to the meeting.",
    "Thanks for helping me with the move this weekend.",
    "What time should we meet at the restaurant?",
    "The weather is supposed to be nice tomorrow.",
    "Can you pick up some coffee on your way here?",
    
    # Longer complex sentences
    "The implementation of the new security protocol requires careful consideration of existing infrastructure limitations.",
    "Despite the challenges we faced during development, the team successfully delivered the project ahead of schedule.",
    "Please ensure all necessary documentation is completed before submitting the final report to the committee.",
    "The integration between the legacy system and new platform should be thoroughly tested before deployment.",
    
    # Common phrases
    "Looking forward to hearing from you soon.",
    "Please find attached the requested information.",
    "Let me know if you need anything else.",
    "I hope this email finds you well.",
    "Thanks for bringing this to my attention.",
    "I appreciate your quick response.",
    
    # Technical instructions
    "Install the required dependencies using the package manager.",
    "Ensure all tests pass before merging the pull request.",
    "The configuration file should be placed in the root directory.",
    "Update the environment variables according to the documentation.",
    "Remember to handle edge cases in the implementation.",
    
    # Project management
    "The sprint planning meeting is scheduled for Monday morning.",
    "Please update your tasks in the project management system.",
    "We need to prioritize the critical bug fixes this week.",
    "The client requested additional features for the next release.",
    "Team availability should be updated in the shared calendar.",
    
    # Error messages and notifications
    "An unexpected error occurred during the process.",
    "Your session has expired, please log in again.",
    "The requested resource is temporarily unavailable.",
    "Invalid credentials, please check your username and password.",
    
    # Meeting notes
    "Action items were assigned to respective team members.",
    "The next review meeting is scheduled for next Thursday.",
    "Key decisions were documented in the shared workspace.",
    "Budget allocations for Q4 were discussed and approved.",
    
    # Customer service
    "We apologize for any inconvenience this may have caused.",
    "Your feedback helps us improve our services.",
    "A support representative will contact you shortly.",
    "Please provide your order number for reference.",
    
    # System messages
    "The system maintenance is scheduled for this weekend.",
    "All users should save their work before the update begins.",
    "Database backup process will start automatically.",
    "Please update your password according to the new policy.",
    
    # Product descriptions
    "This feature enables seamless integration with existing tools.",
    "Advanced analytics provide detailed insights into user behavior.",
    "The new interface offers improved accessibility options.",
    "Regular updates ensure optimal performance and security.",
    
    # Academic writing
    "The research methodology follows established scientific principles.",
    "Results indicate a significant correlation between variables.",
    "Further studies are needed to validate these findings.",
    "The literature review reveals several important gaps.",
    
    # Personal notes
    "Remember to call back about the insurance quote.",
    "Need to finish the presentation by end of day.",
    "Dentist appointment next Tuesday at 2:30.",
    "Pick up groceries on the way home from work.",
    
    # Social media
    "Check out our latest blog post about industry trends.",
    "Don't forget to subscribe to our newsletter.",
    "Share your thoughts in the comments below.",
    "Follow us for more updates and announcements.",
    
    # Long technical explanations
    "The recursive implementation of the algorithm ensures optimal space complexity while maintaining reasonable performance characteristics.",
    "Cross-platform compatibility requires careful consideration of different operating system architectures and their specific constraints.",
    "Microservice architecture enables independent scaling and deployment of individual components while increasing system resilience.",
    
    # Common responses
    "I'll look into this and get back to you soon.",
    "That works perfectly for my schedule.",
    "Could you please clarify what you mean by that?",
    "I'm not available at that time, can we reschedule?",
    
    # Process descriptions
    "First initialize the environment variables before starting the application.",
    "Regularly backup your data to prevent potential loss.",
    "Monitor system resources to ensure optimal performance.",
    "Document any changes made to the production environment."
    ]
    print("Testing different error configurations:\n")
    
    for test_text in example_texts:
        print(f"Original: {test_text}")
        # generate random config
        config = [random.uniform(0, 1) for _ in range(6)]
        typo_generator = TypoGenerator(*config)
        typo_text = typo_generator.generate_typos(test_text)
        print(f"Generated: {typo_text}\n")


if __name__ == "__main__":
    main()