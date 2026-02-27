import React, { useState, useRef, useEffect } from "react";
import {
  Container,
  Card,
  Button,
  Spinner,
  ProgressBar,
} from "react-bootstrap";

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";

import * as mobilenet from "@tensorflow-models/mobilenet";

function App() {
  // Ссылка на загруженное изображение (base64)
  const [imageUrl, setImageUrl] = useState(null);

  // Результаты классификации
  const [predictions, setPredictions] = useState([]);

  const [loading, setLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  const imgRef = useRef(null);

  // Сохраняем модель
  const modelRef = useRef(null);

  // Загруджаем модель при старте приложения
  useEffect(() => {
    const loadModel = async () => {
      // Указываем backend (CPU — самый стабильный)
      await tf.setBackend("cpu");

      // Ждём готовности TensorFlow
      await tf.ready();

      // Загружаем саму MobileNet
      modelRef.current = await mobilenet.load({
        version: 1,
        alpha: 0.25, // самая лёгкая версия
      });

      setModelLoaded(true);
      console.log("Модель загружена");
    };

    loadModel();
  }, []);

  // Загрузка изображения
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    // Сохраняем новое изображение
    reader.onload = () => {
      setImageUrl(reader.result);
      setPredictions([]);
    };

    reader.readAsDataURL(file);
  };

  // Классификация изображения
  const classifyImage = async () => {
    if (!modelRef.current || !imgRef.current) return;

    setLoading(true);

    // Передаём изображение модели
    const results = await modelRef.current.classify(imgRef.current);

    // Берём только top-3 результата
    setPredictions(results.slice(0, 3));

    setLoading(false);
  };

  // Очистка
  const clearAll = () => {
    setImageUrl(null);
    setPredictions([]);
  };

  // UI
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#0d1117",
      }}
    >
      <Container style={{ maxWidth: "480px" }}>
        <Card
          style={{
            padding: "32px",
            borderRadius: "16px",
            backgroundColor: "#161b22",
            color: "white",
            border: "1px solid #30363d",
          }}
        >
          <h4 style={{ textAlign: "center", marginBottom: "24px" }}>
            Image Recognition
          </h4>

          {/* Кнопка загрузки */}
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={{
              marginBottom: "16px",
            }}
          />

          {/* Если изображение выбрано — показываем его */}
          {imageUrl && (
            <>
              <img
                ref={imgRef}
                src={imageUrl}
                alt="preview"
                style={{
                  width: "100%",
                  borderRadius: "12px",
                  marginBottom: "20px",
                  objectFit: "cover",
                }}
              />

              <Button
                onClick={classifyImage}
                disabled={!modelLoaded || loading}
                style={{
                  width: "100%",
                  marginBottom: "10px",
                  backgroundColor: "#238636",
                  border: "none",
                }}
              >
                {loading ? (
                  <Spinner animation="border" size="sm" />
                ) : (
                  "Analyze Image"
                )}
              </Button>

              <Button
                variant="outline-light"
                onClick={clearAll}
                style={{ width: "100%" }}
              >
                Reset
              </Button>
            </>
          )}

          {/* Блок результатов */}
          {predictions.length > 0 && (
            <div style={{ marginTop: "24px" }}>
              {predictions.map((pred, index) => (
                <div key={index} style={{ marginBottom: "16px" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "14px",
                      marginBottom: "6px",
                    }}
                  >
                    <span>{pred.className}</span>
                    <span>
                      {(pred.probability * 100).toFixed(1)}%
                    </span>
                  </div>

                  <ProgressBar
                    now={pred.probability * 100}
                    style={{
                      height: "6px",
                      backgroundColor: "#30363d",
                    }}
                  />
                </div>
              ))}
            </div>
          )}
        </Card>
      </Container>
    </div>
  );
}

export default App;