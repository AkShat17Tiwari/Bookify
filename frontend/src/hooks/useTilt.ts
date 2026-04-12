import { useState, useCallback, useRef } from 'react';

export function useTilt(maxTilt: number = 8) {
  const [style, setStyle] = useState({ rotateX: 0, rotateY: 0 });
  const ref = useRef<HTMLDivElement>(null);

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -maxTilt;
      const rotateY = ((x - centerX) / centerX) * maxTilt;
      setStyle({ rotateX, rotateY });
    },
    [maxTilt]
  );

  const onMouseLeave = useCallback(() => {
    setStyle({ rotateX: 0, rotateY: 0 });
  }, []);

  return { ref, style, onMouseMove, onMouseLeave };
}
