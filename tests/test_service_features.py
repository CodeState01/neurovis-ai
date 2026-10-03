"""Validation tests for safe growth and local image preprocessing."""
import base64
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from student import service
from PIL import Image

class ServiceFeatureTests(unittest.TestCase):
    def test_lesson_width_is_bounded_to_256(self):
        examples=[{'question':f'Pergunta número {i}?','answer':'Resposta breve.'} for i in range(12)]
        lesson=service.validate_lesson({'width':256,'steps':40,'examples':examples},8,40,True)
        self.assertEqual(lesson['width'],256)
        with self.assertRaises(ValueError):
            service.validate_lesson({'width':257,'steps':40,'examples':examples},8,40,True)

    def test_vision_reencodes_image_and_uses_installed_local_teacher(self):
        buf=io.BytesIO();Image.new('RGB',(40,30),(0,20,255)).save(buf,format='PNG')
        seen={}
        def fake_ollama(path,payload,timeout=0):
            seen.update(payload=payload,timeout=timeout)
            return {'message':{'content':'A imagem mostra um quadrado azul.'}}
        with patch.object(service,'ollama',side_effect=fake_ollama):
            result=service.vision_description('Qual é a cor do quadrado?',base64.b64encode(buf.getvalue()).decode(),'image/png')
        self.assertEqual(result,'A imagem mostra um quadrado azul.')
        self.assertEqual(seen['payload']['model'],'qwen3.5:4b')
        self.assertEqual(len(seen['payload']['messages'][0]['images']),1)
        self.assertEqual(seen['payload']['options']['num_predict'],300)

    def test_vision_rejects_bad_or_oversized_image_data(self):
        with self.assertRaises(ValueError):service.vision_description('descreva','não é base64','image/jpeg')
        with self.assertRaises(ValueError):service.vision_description('descreva','a'*8_000_001,'image/jpeg')

if __name__=='__main__':unittest.main()
