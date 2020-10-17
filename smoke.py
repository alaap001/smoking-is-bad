from fastai.vision import * 
from bottle import route, run,request,response
import PIL
# data2 = ImageDataBunch.single_from_classes(
#     path, data.classes, tfms=tfms, size=224).normalize(imagenet_stats)
# learn = create_cnn(data2, models.resnet34)
# learn.load('one-epoch')


@route('/predict_image')
def predict_image(mult_img=False):
	response_data={}
	response.status = 400

	params = dict(request.GET)

	img_url = params.get("img_url", "")

	try:
		file = requests.get(img_url, stream=True)
		img = PIL.Image.open(BytesIO(file.content)).convert("RGB")
	except Exception as e:
		response_data['result'] = "Result can't be shown, Image not exist!" + str(e)
		response_data['status'] = -1
		return response_data
	print("image Accepted")
	
	try:
		img_fastai = Image(pil2tensor(img, dtype=np.float32).div_(255))	
		learn = load_learner(os.getcwd(),'smoke_v1.pkl')
		
		if mult_img:
			path = Path(im)
			files = get_image_files(path)
			preds,y = learn.get_preds(files)
			
			response_data['result'] = preds
		else:
			cat,_,pred = learn.predict(img_fastai)
			
			response_data['result'] = "Category is {0} with probability of {1:.2f}%".format(cat,max(pred)*100)
			response_data['score'] = "{0:.2f}".format(max(pred)*100) 
			response_data['status']=1

	except Exception as e:
		response_data['result'] = "Result can't be shown, Image not exist!" + str(e)
		response_data['status'] = -1
	response.status=200
	return response_data


if __name__ == '__main__':

	# im = r'C:\Users\wwech\Desktop\windows\final_again\val\hookah\image3.jpeg'
	# img_url='https://b1.pngbarn.com/png/805/998/mon3y-set-lighted-cigarette-png-clip-art-thumbnail.png'
	
	# print(predict_image())
	run(reloader=True)