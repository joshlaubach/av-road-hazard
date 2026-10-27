"use client";

import { useState, type ChangeEvent, type DragEvent } from "react";

interface UploaderProps {
  onFile: (file: File) => void;
  isBusy: boolean;
}

/** Pick or drop a photo. The whole box is one label around a real file input, so it works by keyboard. */
export function Uploader({ onFile, isBusy }: UploaderProps) {
  const [isOver, setIsOver] = useState(false);

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) onFile(file);
    event.target.value = ""; // so the same photo can be chosen again
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsOver(false);
    const file = event.dataTransfer.files[0];
    if (file) onFile(file);
  }

  return (
    <label
      className={`dropzone${isOver ? " is-over" : ""}${isBusy ? " is-busy" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsOver(true);
      }}
      onDragLeave={() => setIsOver(false)}
      onDrop={handleDrop}
    >
      <input type="file" accept="image/jpeg,image/png" onChange={handleChange} disabled={isBusy} />
      <span className="dropzone-title">Choose a road photo</span>
      <span className="dropzone-hint">or drop one here. JPEG or PNG, up to 10 megabytes.</span>
    </label>
  );
}
