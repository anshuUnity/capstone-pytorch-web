from django.shortcuts import render
from django.http import JsonResponse
from .models import Prediction
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from calory_counter import settings
import torch
from torchvision import transforms, models
import openai
from PIL import Image
import json

# Load the model
model = models.resnet50()

# Adjust the fully connected layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 101)  # Change 101 to match the number of classes in your saved model

# Load the state dict
model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=torch.device('cpu')))

# Set model to evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device("cpu")
model = model.to(device)

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Food-101 classes
FOOD101_CLASSES = ['Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 
                   'Beef Tartare', 'Beet Salad', 'Beignets', 'Bibimbap', 'Bread Pudding', 
                   'Breakfast Burrito', 'Bruschetta', 'Caesar Salad', 'Cannoli', 'Caprese Salad', 
                   'Carrot Cake', 'Ceviche', 'Cheesecake', 'Cheese Plate', 'Chicken Curry', 
                   'Chicken Quesadilla', 'Chicken Wings', 'Chocolate Cake', 'Chocolate Mousse', 'Churros', 'Clam Chowder', 
                   'Club Sandwich', 'Crab Cakes', 'Creme Brulee', 'Croque Madame', 'Cup Cakes', 'Deviled Eggs', 'Donuts', 
                   'Dumplings', 'Edamame', 'Eggs Benedict', 'Escargots', 'Falafel', 'Filet Mignon', 'Fish And Chips', 
                   'Foie Gras', 'French Fries', 'French Onion Soup', 'French Toast', 'Fried Calamari', 'Fried Rice', 
                   'Frozen Yogurt', 'Garlic Bread', 'Gnocchi', 'Greek Salad', 'Grilled Cheese Sandwich', 'Grilled Salmon', 
                   'Guacamole', 'Gyoza', 'Hamburger', 'Hot And Sour Soup', 'Hot Dog', 'Huevos Rancheros', 'Hummus', 'Ice Cream', 
                   'Lasagna', 'Lobster Bisque', 'Lobster Roll Sandwich', 'Macaroni And Cheese', 'Macarons', 'Miso Soup', 'Mussels', 
                   'Nachos', 'Omelette', 'Onion Rings', 'Oysters', 'Pad Thai', 'Paella', 'Pancakes', 'Panna Cotta', 'Peking Duck', 
                   'Pho', 'Pizza', 'Pork Chop', 'Poutine', 'Prime Rib', 'Pulled Pork Sandwich', 'Ramen', 'Ravioli', 'Red Velvet Cake', 
                   'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed Salad', 'Shrimp And Grits', 'Spaghetti Bolognese', 
                   'Spaghetti Carbonara', 'Spring Rolls', 'Steak', 'Strawberry Shortcake', 'Sushi', 'Tacos', 'Takoyaki', 
                   'Tiramisu', 'Tuna Tartare', 'Waffles']

def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return classes[predicted.item()]


# Create your views here.

def index(request):
    return render(request, "home.html")

def upload_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        if image:
            # Save the image to the media directory
            upload_image = Prediction(image=image)
            upload_image.save()
            path = upload_image.image.path
            url = upload_image.image.url

            prediction = predict_image(path, model, transform, FOOD101_CLASSES)
            print(prediction)
            upload_image.predction = prediction
            upload_image.save()
            # image_name = default_storage.save(f'uploaded_images/{image.name}', ContentFile(image.read()))
            # image_url = default_storage.url(image_name)
            
            response_data = {
                'success': True,
                'image_url': url,
                "prediction":prediction
            }
            return JsonResponse(response_data)
        else:
            return JsonResponse({'success': False, 'error': 'No image provided'})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid request method'})
    
def get_more_info(request):
    dish = request.GET.get('dish')
    if not dish:
        return JsonResponse({'success': False, 'error': 'No dish name provided'})

    openai.api_key = settings.OPEN_AI_KEY  # Ensure you set your OpenAI API key as an environment variable

    prompt = f"Provide the following information about the dish '{dish}': Calories, Fun Fact, History,Ingredients and process(list) to make it. Give the response in json format with keys as calories, fun_fact, history, ingredients, process. Give response in a valid json string format, that is response should not contain new line or these types of invalid escape character"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print(type(response.choices[0].message))
        message = response.choices[0].message.content
        print(type(message), "MESSAGE", message)
        json_message = json.loads(message)
        print(json_message, type(json_message), "JSON")
        # info = response['choices'][0]['message']['content'].strip().split('\n')
        
        # Assuming the response is well-structured, we split it into parts
        # calories = info[0].replace("Calories: ", "")
        # fun_fact = info[1].replace("Fun Fact: ", "")
        # history = info[2].replace("History: ", "")
        # ingredients = info[3].replace("Ingredients: ", "")
        
        response_data = json_message
        response_data['success'] = True,
        #     'calories': json_message['calories'],
        #     'fun_fact': json_message['fun_fact'],
        #     'history': json_message['history'],
        #     'ingredients': json_message['ingredients']
        # }
    except Exception as e:
        response_data = {
            'success': False,
            'error': str(e)
        }

    return JsonResponse(response_data)