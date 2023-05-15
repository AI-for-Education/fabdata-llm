from typing import Dict, List, Union
import pickle

from .llmtypes import LLMMessage

class LLMMessageCache:
    def __init__(self) -> None:
        self.Cache_Items: Dict[str, List[LLMMessage]] = {}
        self.max_user_cache_items = 100

    def __getitem__(self, phone_number: str) -> List[LLMMessage]:
        return self.get_user_chat_history(phone_number)

    def get_user_chat_history(self, phone_number: str):
        convo_history = self.Cache_Items.get(phone_number)
        
        if convo_history:
            return convo_history
        return []
    
    def get_user_chat_history_last_entry(self, phone_number: str):
        convo_history = self.Cache_Items.get(phone_number)
        
        if convo_history:
            if len(convo_history) > 0:
                return convo_history[len(convo_history) - 1]
            
        return None
    
    def add_item(self, phone_number: str, new_item: LLMMessage):
        self.add_items(phone_number, [new_item])

    def add_items(self, phone_number: str, new_items: List[LLMMessage]):
        convo_history = self.__getitem__(phone_number)
        
        if convo_history:
            convo_history.extend(new_items)

            if len(convo_history) > self.max_user_cache_items:
                convo_history = convo_history[2:]

            self.Cache_Items[phone_number] = convo_history
        else:
            self.Cache_Items[phone_number] = new_items

    def reset_cache(self, phone_number:str):
        convo_history = self.__getitem__(phone_number)
        
        if convo_history:
            self.Cache_Items[phone_number] = []

