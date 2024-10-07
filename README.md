# Llama3Finetuning

1. DataProcess
   (1) run callOPenAi.py to extract entities from given text. (description.txt->ExtractionResult.txt)
   (2) run comebineData.py to comebine the text and extrated entities into pairs and save in json. (description.txt + ExtractionResult.txt -> CustomData.json)
  
3. run finetune.py
