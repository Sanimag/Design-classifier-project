from django.shortcuts import render
# Create your views here.
def Main(request):
	return render(request, 'index.html')

def Result(request):
	product_description = request.POST['product_description']
	return render(request, 'result.html', {'description':product_description})
