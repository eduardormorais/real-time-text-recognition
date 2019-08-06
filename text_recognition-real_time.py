from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import pytesseract
import cv2

def decode_predictions(scores, geometry):	
	# Pegua o número de linhas e colunas do volume de pontuações, então inicializa o conjunto de retângulos de caixa delimitadoras e correspondentes a pontuações de confiança
	(num_rows, num_cols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, num_rows):
		# Extrair as pontuações (probabilidades), seguidas da dados geométricos usados ​​para derivar a caixa delimitadora potencial coordenadas que circundam o texto
		scores_data = scores[0, 0, y]
		x_data_0 = geometry[0, 0, y]
		x_data_1 = geometry[0, 1, y]
		x_data_2 = geometry[0, 2, y]
		x_data_3 = geometry[0, 3, y]
		angles_data = geometry[0, 4, y]

		for x in range(0, num_cols):
			# Se a pontuação não tiver uma probabilidade suficiente, ignora
			if scores_data[x] < 0.9:
				continue

			# Calcula o fator de offset como recurso resultante maps que será 4x menor que a imagem de entrada
			(offset_x, offset_y) = (x * 4.0, y * 4.0)

			# Extrai o ângulo de rotação para a previsão e então calcula o seno e cosseno
			angle = angles_data[x]
			cos = np.cos(angle)
			sin = np.sin(angle)


			# Usa o volume da geometria para obter a largura e a altura da caixa delimitadora
			h = x_data_0[x] + x_data_2[x]
			w = x_data_1[x] + x_data_3[x]

			# Calcula as coordenadas inicial e final (x, y) para a caixa delimitadora de previsão de texto
			end_x = int(offset_x + (cos * x_data_1[x]) + (sin * x_data_2[x]))
			end_y = int(offset_y - (sin * x_data_1[x]) + (cos * x_data_2[x]))
			start_x = int(end_x - w)
			start_y = int(end_y - h)
		
			# Adicione as coordenadas da caixa delimitadora e o escore de probabilidade para as nossas respectivas listas
			rects.append((start_x, start_y, end_x, end_y))
			confidences.append(scores_data[x])

	# Retorna uma tupla das caixas delimitadoras e confidências associadas
	return (rects, confidences)

def initialize_dimension(width, height):
	(W, H) = (None, None)
	(new_w, new_h) = (width, height)
	(r_w, r_h) = (None, None)	
	return H, W, r_h, r_w, new_w, new_h

def resizing(frame, H, W, r_w, r_h):
	(H, W) = frame.shape[:2]
	r_w = W / float(new_w)
	r_h = H / float(new_h)
	return H, W, r_w, r_h

def model_detect_boxes(model, frame, new_w, new_h):
	# Construir um blob a partir do quadro e, em seguida, executar um passe para frente do modelo para obter os dois conjuntos de camadas de saída
	blob = cv2.dnn.blobFromImage(frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
	model.setInput(blob)
	(scores, geometry) = model.forward(layer_names)
	# Decodifica as previsões e aplica a supressão não máxima para suprimir caixas delimitadoras fracas e sobrepostas
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	return boxes

def detect_character_from_boxes(boxes, r_w, r_h, dic_results):	
	# loop sobre as caixas delimitadoras
	for (start_x, start_y, end_x, end_y) in boxes:
		# Dimensionar as coordenadas da caixa delimitadora com base nos respectivos resultados
		start_x = int(start_x * r_w)
		start_y = int(start_y * r_h)
		end_x = int(end_x * r_w)
		end_y = int(end_y * r_h)

		roi = orig[start_y:end_y, start_x:end_x]
		cv2.imshow("Roi", roi)
		
		# Reconhece characters da imagem
		dic_results = text_detection(roi, orig, dic_results, start_x, start_y, end_x, end_y)
		
def text_detection(roi, orig, dic_results, start_x, start_y, end_x, end_y):
	#Configuração para leitura de caracteres pytesseract
	config = ("-l eng+por --oem 1 --psm 7")
	
	text = pytesseract.image_to_string(roi, config=config)
	
	if text is not "":
		text = pytesseract.image_to_string(roi, config=config)
		if text in dic_results:
			dic_results[text] += 1
			if dic_results[text] > 5:
				print("Texto: %s Qtd: %d" %(text, dic_results[text]))
				text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
				# draw the bounding box on the frame
				cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
				cv2.putText(orig, text, (start_x, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
		else:
			dic_results[text] = 1
	
	return dic_results

if __name__ == "__main__":

	# Inicializa as dimensões originais do quadro, as novas dimensões e as relações entre as dimensões
	height_H, width_W, r_h, r_w, new_w, new_h = initialize_dimension(160, 160)

	# Define os dois nomes de camadas de saída para o modelo ​​- o primeiro é as probabilidades de saída e o segundo pode ser usado para derivar as coordenadas da caixa delimitadora do texto
	layer_names = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

	# Carregar o modelo EAST pré-treinado
	model = cv2.dnn.readNetFromTensorflow("frozen_east_text_detection.pb")

	# Pegua a referência para a web cam
	vs = cv2.VideoCapture(0)

	# Iniciar o estimador de taxa de transferência do FPS
	fps = FPS().start()

	# loop em quadros do fluxo de vídeo
	cont = 10000
	dic_results = {}

	while True:
		if cont > 10000:
			# Pegua o quadro atual
			frame = vs.read()
			frame = frame[1]

			if frame is None:
				break

			# Redimensionar o quadro, mantendo a proporção
			frame = imutils.resize(frame, width=800)
			orig = frame.copy()
			
			# Se nossas dimensões de quadro forem Nenhum, camptura proporção de dimensões de quadro antigas para novas dimensões de quadro
			if width_W is None or height_H is None:
				height_H, width_W, r_w, r_h = resizing(frame, height_H, width_W, r_w, r_h)

			# Redimensionar o quadro
			frame = cv2.resize(frame, (new_w, new_h))
			
			boxes = model_detect_boxes(model, frame, new_w, new_h)
			
			detect_character_from_boxes(boxes, r_w, r_h, dic_results)
			
			fps.update()

			cv2.imshow("Text Detection", orig)
			key = cv2.waitKey(1) & 0xFF

			# Se a tecla `q` foi pressionada, interrompa o loop
			if key == ord("q"):
				break
			
			cont = 0
		else:
			#print(cont)
			cont+=1

	# Pare o cronômetro e exiba as informações do FPS
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()
