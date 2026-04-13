# import requests
# url = "http://35.220.164.252:3888/v1/chat/completions"
# body = """{
#   "model": "qwen3.5-plus",
#   "messages": [
#     {
#       "role": "system",
#       "content": "请介绍一下你自己"
#     }
#   ]
# }"""
# response = requests.request("POST", url, data = body, headers = {
#   "Content-Type": "application/json", 
#   "Authorization": "sk-SrQRDlXzOIgQz9ciSq9SABqkOFmpGJHQO3BU95Bo01ap63VH "
# })
# print(response.text)

import httpx, time                                                                                                                                           
from core.config import API_KEY, BASE_URL, MODEL_NAME                                                                                                        
                                                                                                                                                            
print(f'模型: {MODEL_NAME}')                                                                                                                                 
client = httpx.Client(trust_env=False, timeout=60)                                                                                                           
t0 = time.time()                                                                                                                                             
resp = client.post(                                                                                                                                          
    f'{BASE_URL}/chat/completions',                                                                                                                          
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},                                                                      
    json={                                                                                                                                                   
        'model': MODEL_NAME,                                                                                                                                 
        'messages': [{'role': 'user', 'content': '你是什么模型，报出你的具体型号，不要只说自己的公司名称'}],                                                                                     
        'max_tokens': 50,                                                                                                                                    
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},                                                                                  
    }                                                                                                                                                        
)                                                                                                                                                            
elapsed = time.time() - t0                                                                                                                                   
data = resp.json()                                                                                                                                           
content = data.get('choices', [{}])[0].get('message', {}).get('content', data)                                                                               
print(f'耗时: {elapsed:.1f}s')                                                                                                                               
print(f'回复: {content}') 